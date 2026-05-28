"""calculate_gflops.py

Estimation of inference-time multiply–accumulate operations (MACs) and
floating-point operations (FLOPs) for DETR-family object-detection models,
including variants in which the transformer encoder is replaced by Mamba-2
selective state-space blocks (Dao & Gu, 2024).

Methodology
-----------
The procedure is hybrid. Standard layers (convolutions, linear projections,
multi-head attention, normalisation) are counted by `fvcore.FlopCountAnalysis`,
which performs a `torch.jit.trace` of the model and dispatches per-operator
handlers on the resulting graph. Because the Mamba-2 SSD (structured
state-space duality) kernel is implemented in Triton, and `torch.jit.trace`
substitutes Python integers with traced symbolic values that the Triton
compiler cannot accept, a direct trace of a model containing
`mamba_ssm.modules.mamba2.Mamba2` fails inside the kernel
`_chunk_cumsum_fwd_kernel` with `IncompatibleTypeErrorImpl: invalid operands
of type pointer<int64> and triton.language.int32`. This failure occurs in
the trace itself and cannot be resolved by operator-handler registration.

The resolution adopted here is to replace each `Mamba2` instance, for the
duration of FLOP counting only, with a traceable proxy module
(`Mamba2TraceProxy`) that retains the genuine input and output linear
projections — so their MACs are still counted by `fvcore` — and substitutes
the SSD kernel with a shape-preserving but FLOP-free operation. The MACs of
the SSD scan are then supplied analytically (see `mamba2_ssd_macs`) and
added to the traced total.

Convention
----------
`fvcore` counts one fused multiply–add as a single FLOP, equivalently a
single MAC. Reported FLOPs are therefore obtained as `2 * MACs`. Both
quantities are emitted to standard output and to Weights & Biases.
"""

import argparse
import contextlib
from pathlib import Path
from typing import Dict, Tuple

import torch
import wandb
from codecarbon import track_emissions
from fvcore.nn import (
    FlopCountAnalysis,
    flop_count_table,
    parameter_count_table,
)
from torch import nn

from jormungandr.config.configuration import (
    FafnirConfig,
    JormungandrConfig,
    WANDB_API_KEY,
    WANDB_ENTITY,
    WANDB_PROJECT,
    load_config,
)
from jormungandr.fafnir import Fafnir
from jormungandr.jormungandr import Jormungandr
from jormungandr.utils.seed import seed_everything

# The Mamba-2 module is imported optionally so that this script remains
# usable for models (e.g. baseline `Jormungandr`) that do not depend on it.
try:
    from mamba_ssm.modules.mamba2 import Mamba2
except ImportError:  # pragma: no cover
    Mamba2 = None  # type: ignore[assignment, misc]


# =====================================================================
# 1. Analytical FLOP estimation for the Mamba-2 SSD scan
# =====================================================================

def mamba2_ssd_macs(
    *,
    B: int,
    L: int,
    d_inner: int,
    d_state: int,
    ngroups: int,
    d_conv: int,
    chunk_size: int,
) -> int:
    """Multiply–accumulate count for a single Mamba-2 block, excluding the
    in_proj and out_proj linear projections (which are counted by fvcore on
    the proxy).

    Notation follows Dao & Gu (2024):

        D      = d_model
        d_inner = E · D  (typically E = 2)
        N      = d_state                (default 128)
        H      = d_inner / d_head       (default d_head = 64)
        G      = ngroups                (default 1)
        K      = d_conv                 (default 4)
        C      = chunk_size             (default 256)

    The dominant terms are

        conv1d              :  B · L · (d_inner + 2·G·N) · K
        intra-chunk SSD     :  B · L · C · d_inner · G
        inter-chunk states  :  B · L · d_inner · N
        state-to-output     :  B · L · d_inner · N

    The constant factor on the SSD terms differs across derivations; the
    above expressions agree with the asymptotic form O(L · d_inner · N)
    stated in the Mamba-2 paper, and serve as a defensible lower-bound
    approximation. The formula is reported verbatim in the methodology
    section of the thesis.
    """
    conv = B * L * (d_inner + 2 * ngroups * d_state) * d_conv
    intra_chunk = B * L * chunk_size * d_inner * ngroups
    inter_chunk = B * L * d_inner * d_state
    state_output = B * L * d_inner * d_state
    return conv + intra_chunk + inter_chunk + state_output


# =====================================================================
# 2. Trace-compatible proxy for Mamba2
# =====================================================================

class Mamba2TraceProxy(nn.Module):
    """Drop-in replacement for `Mamba2` used exclusively for FLOP counting.

    The proxy retains references to the original `in_proj` and `out_proj`
    submodules of the source `Mamba2` block, so that `fvcore` counts their
    MACs against the proper module path. The Triton-resident SSD kernel is
    replaced by a zero tensor of the appropriate shape, which is traceable
    and contributes no FLOPs. The shape of the input tensor observed during
    the traced forward pass is recorded in `last_input_shape` for later
    use by `mamba2_ssd_macs`.
    """

    def __init__(self, source: "Mamba2") -> None:
        super().__init__()
        # Reuse the actual parameter tensors so that parameter counts and
        # projection MACs are identical to those of the deployed model.
        self.in_proj = source.in_proj
        self.out_proj = source.out_proj

        # Architectural hyperparameters required for analytical counting.
        self.d_model = source.d_model
        self.d_inner = source.d_inner
        self.d_state = source.d_state
        self.ngroups = source.ngroups
        self.d_conv = source.d_conv
        self.chunk_size = source.chunk_size

        # Populated by `forward`; consumed after the trace completes.
        self.last_input_shape: Tuple[int, ...] | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, d_model). The output must depend on `proj` so that
        # torch.jit.trace's dead-code elimination retains the in_proj call.
        # We slice the first d_inner channels of the projection — matching
        # out_proj.in_features — and scale by zero to neutralise the value
        # while preserving the trace dependency.
        self.last_input_shape = tuple(x.shape)
        proj = self.in_proj(x)                                # counted by fvcore
        zeroed = proj[..., : self.out_proj.in_features] * 0.0  # preserves trace edge
        return self.out_proj(zeroed)                          # counted by fvcore


@contextlib.contextmanager
def swap_mamba_for_proxy(model: nn.Module):
    """Context manager that replaces every `Mamba2` submodule of `model`
    with a `Mamba2TraceProxy` for the duration of the `with` block, and
    restores the originals on exit. Yields a dictionary mapping qualified
    module names to (original, proxy) pairs.
    """
    if Mamba2 is None:
        yield {}
        return

    swaps: Dict[str, Tuple["Mamba2", Mamba2TraceProxy]] = {}
    for name, module in list(model.named_modules()):
        if isinstance(module, Mamba2):
            proxy = Mamba2TraceProxy(module)
            parent_path, _, leaf = name.rpartition(".")
            parent = model.get_submodule(parent_path) if parent_path else model
            setattr(parent, leaf, proxy)
            swaps[name] = (module, proxy)
    try:
        yield swaps
    finally:
        for name, (original, _) in swaps.items():
            parent_path, _, leaf = name.rpartition(".")
            parent = model.get_submodule(parent_path) if parent_path else model
            setattr(parent, leaf, original)


# =====================================================================
# 3. Operator handlers for fvcore
# =====================================================================

def _sdpa_macs(inputs, outputs) -> int:
    """fvcore handler for `aten::scaled_dot_product_attention`.

    The fused SDPA operator is not recognised by fvcore's default operator
    set. For inputs of shape (..., L_q, D_q) (query) and (..., L_k, D_k)
    (key/value), the operation comprises two matrix multiplications:

        Q @ K^T :  prod(batch_dims) * L_q * D_q * L_k
        attn @ V:  prod(batch_dims) * L_q * L_k * D_v

    Assuming D_q = D_v = D (the standard configuration), the total MAC
    count is 2 * prod(batch_dims) * L_q * L_k * D.
    """
    q_shape = inputs[0].type().sizes()
    k_shape = inputs[1].type().sizes()
    *q_batch, L_q, D = q_shape
    *_, L_k, _ = k_shape
    batch = 1
    for dim in q_batch:
        batch *= dim
    return 2 * batch * L_q * L_k * D


# =====================================================================
# 4. Composite FLOP analysis routine
# =====================================================================

def calculate_gflops(model: nn.Module, input_size: Tuple[int, int, int, int]) -> None:
    """Compute and report the inference MAC/FLOP count of `model` on a
    synthetic input of shape `input_size`. The traced contribution and the
    analytical SSD contribution are reported separately and as a sum.
    """
    model.eval()
    dummy = torch.randn(*input_size).cuda()

    with swap_mamba_for_proxy(model) as swaps, torch.no_grad():
        flops = FlopCountAnalysis(model, dummy)
        flops.set_op_handle("aten::scaled_dot_product_attention", _sdpa_macs)
        traced_macs = flops.total()
        unsupported = dict(flops.unsupported_ops())
        uncalled = list(flops.uncalled_modules())
        traced_table = flop_count_table(flops, max_depth=3)
        by_module = dict(flops.by_module())
        by_operator = dict(flops.by_operator())

    # ----- Analytical SSD contribution -----
    ssd_macs_total = 0
    per_block_ssd: Dict[str, int] = {}
    missing_blocks = []
    for name, (_, proxy) in swaps.items():
        if proxy.last_input_shape is None:
            missing_blocks.append(name)
            continue
        B, L, _ = proxy.last_input_shape
        block_macs = mamba2_ssd_macs(
            B=B,
            L=L,
            d_inner=proxy.d_inner,
            d_state=proxy.d_state,
            ngroups=proxy.ngroups,
            d_conv=proxy.d_conv,
            chunk_size=proxy.chunk_size,
        )
        ssd_macs_total += block_macs
        per_block_ssd[name] = block_macs

    total_macs = traced_macs + ssd_macs_total
    param_table = parameter_count_table(model)

    # ----- Reporting -----
    print(traced_table)
    print(param_table)
    print(f"Mamba-2 blocks detected           : {len(swaps)}")
    if missing_blocks:
        print(f"  WARNING: blocks not invoked     : {missing_blocks}")
    print(f"Traced MACs (fvcore, excl. SSD)    : {traced_macs / 1e9:>10.3f} G")
    print(f"Analytical Mamba-2 SSD MACs        : {ssd_macs_total / 1e9:>10.3f} G")
    print(f"Total MACs                          : {total_macs / 1e9:>10.3f} G")
    print(f"Total FLOPs (= 2 x MACs)            : {2 * total_macs / 1e9:>10.3f} G")
    print(f"Unsupported ops in traced subgraph : {unsupported}")
    print(f"Uncalled modules                    : {uncalled}")

    # ----- Logging -----
    wandb.log(
        {
            "gflops/traced_macs":       traced_macs / 1e9,
            "gflops/ssd_analytical":    ssd_macs_total / 1e9,
            "gflops/total_macs":        total_macs / 1e9,
            "gflops/total_flops_2x":    2 * total_macs / 1e9,
            "gflops/n_mamba_blocks":    len(swaps),
            "gflops/n_uncalled":        len(uncalled),
            "gflops/unsupported_ops":   unsupported,
            "gflops/by_module":         by_module,
            "gflops/by_operator":       by_operator,
            "gflops/ssd_per_block":     {k: v / 1e9 for k, v in per_block_ssd.items()},
        }
    )


# =====================================================================
# 5. Entry point
# =====================================================================

@track_emissions(
    country_iso_code="NOR",
    project_name="fafnir_training",
    log_level="ERROR",
)
def main(config_file: str) -> None:
    config = load_config(config_file)
    seed_everything(config.trainer.seed)

    wandb.login(key=WANDB_API_KEY)
    wandb.init(
        project=WANDB_PROJECT,
        entity=WANDB_ENTITY,
        name=f"gflops_{Path(config_file).stem}",
        # mode="disabled",
        config=config.model_dump(),
    )
    device = "cuda"

    if isinstance(config.model, JormungandrConfig):
        model = Jormungandr(config=config.model).to(device)
    elif isinstance(config.model, FafnirConfig):
        model = Fafnir(config=config.model).to(device)
    else:
        raise TypeError(f"Unsupported model configuration: {type(config.model)}")

    calculate_gflops(model, input_size=(1, 3, 800, 1333))
    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    experiment = "experiment_2.yaml"
    parser.add_argument(
        "config",
        nargs="?",
        default=None,
        help=f"Config file to load (e.g. {experiment})",
    )
    parser.add_argument(
        "--config",
        dest="config_flag",
        default=None,
        help=f"Config file to load (e.g. {experiment})",
    )
    parser.add_argument(
        "--model-path",
        dest="model_path",
        default=None,
        help="Path to the trained model checkpoint to validate",
    )
    args = parser.parse_args()
    main(args.config_flag or args.config or experiment)