"""
Sweep inference FPS and peak GPU memory across image resolutions for Fafnir.

For each square resolution R, synthetic flattened feature maps of shape
(B, (R/stride)^2, D) are constructed and fed directly through the encoder.
The backbone, decoder, and prediction head are bypassed so that the
reported latency and peak memory reflect the encoder's scaling behaviour
in isolation. On OOM, log one OOM marker row and stop the sweep (every
larger R will also OOM).
"""

import argparse
import csv
import os
from pathlib import Path

import torch
import wandb
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


DEFAULT_RESOLUTIONS = [256, 384, 512, 640, 768, 896, 1024, 1280, 1536, 1792, 2048, 2560, 3072, 4096, 5120, 6144, 8192, 10240, 12288, 16384, 20480]
DEFAULT_BATCH_SIZE = 8
WARMUP_ITERS = 3
TIMED_ITERS = 10
# Spatial stride of the ResNet-50 backbone used by DETR. The encoder sees
# feature maps of side length R / BACKBONE_STRIDE for an R x R input image.
BACKBONE_STRIDE = 32
RESULTS_DIR = Path("results/benchmark")


def time_resolution(
    model: nn.Module,
    resolution: int,
    batch_size: int,
    model_dimension: int,
    device: str = "cuda",
) -> tuple[float, float, float, float]:
    """Return (mean_latency_ms, fps, peak_mem_mb, encoder_mem_mb) for one resolution.

    Only the encoder is exercised. The function constructs:
      - flattened feature maps      ~ (B, (R/stride)^2, D)
      - a position embedding        via model.embedder
      - an all-False flattened mask ~ (B, (R/stride)^2)
    and feeds these directly into model.encoder.forward, matching exactly the
    tensor shapes that Fafnir.forward assembles immediately prior to the
    encoder call. Peak GPU memory is reported both in absolute terms and as
    the increment over the parameter baseline; the latter isolates the
    encoder's activation memory.

    Raises torch.cuda.OutOfMemoryError if the resolution does not fit.
    """
    torch.cuda.empty_cache()
    baseline_mem = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()

    feature_h = resolution // BACKBONE_STRIDE
    feature_w = resolution // BACKBONE_STRIDE
    seq_len = feature_h * feature_w

    flattened_features = torch.randn(
        batch_size, seq_len, model_dimension, device=device
    )
    mask = torch.zeros(
        batch_size, feature_h, feature_w, dtype=torch.bool, device=device
    )
    position_embedding = model.embedder.forward(
        shape=(batch_size, model_dimension, feature_h, feature_w),
        device=device,
        dtype=flattened_features.dtype,
        mask=mask,
    )
    flattened_mask = mask.flatten(1)

    def run_encoder() -> None:
        model.encoder.forward(
            flattened_features,
            position_embedding=position_embedding,
            pixel_mask=flattened_mask,
        )

    with torch.no_grad():
        for _ in range(WARMUP_ITERS):
            run_encoder()
        torch.cuda.synchronize()

        latencies_ms: list[float] = []
        for _ in range(TIMED_ITERS):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            run_encoder()
            end.record()
            torch.cuda.synchronize()
            latencies_ms.append(start.elapsed_time(end))

    mean_latency_ms = sum(latencies_ms) / len(latencies_ms)
    fps = (batch_size * 1000.0) / mean_latency_ms
    peak_mem_mb = torch.cuda.max_memory_allocated() / 1e6
    encoder_mem_mb = (torch.cuda.max_memory_allocated() - baseline_mem) / 1e6
    return mean_latency_ms, fps, peak_mem_mb, encoder_mem_mb


def main(
    config_file: str,
    resolutions: list[int],
    batch_size: int,
) -> None:
    config = load_config(config_file)
    seed_everything(config.trainer.seed)

    config_stem = Path(config_file).stem
    model_dimension = config.model.model_dimension

    wandb.login(key=WANDB_API_KEY)
    wandb.init(
        project=WANDB_PROJECT,
        entity=WANDB_ENTITY,
        name=f"benchmark_resolution_{config_stem}",
        tags=["benchmark", "resolution", "encoder_only"],
        config={
            **config.model_dump(),
            "benchmark": {
                "batch_size": batch_size,
                "resolutions": resolutions,
                "warmup_iters": WARMUP_ITERS,
                "timed_iters": TIMED_ITERS,
                "backbone_stride": BACKBONE_STRIDE,
                "scope": "encoder_only",
            },
        },
    )

    device = "cuda"
    if isinstance(config.model, JormungandrConfig):
        model: nn.Module = Jormungandr(config=config.model).to(device)
    elif isinstance(config.model, FafnirConfig):
        model = Fafnir(config=config.model).to(device)
    else:
        raise ValueError(f"Unsupported model config type: {type(config.model)}")
    model.eval()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = RESULTS_DIR / f"{config_stem}.csv"

    fieldnames = [
        "config",
        "resolution",
        "seq_len",
        "batch_size",
        "latency_ms",
        "fps",
        "peak_mem_mb",
        "encoder_mem_mb",
        "oom",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        f.flush()

        for resolution in resolutions:
            seq_len = (resolution // BACKBONE_STRIDE) ** 2
            print(
                f"[{config_stem}] resolution={resolution} seq_len={seq_len} ...",
                flush=True,
            )
            try:
                latency_ms, fps, peak_mem_mb, encoder_mem_mb = time_resolution(
                    model, resolution, batch_size, model_dimension, device
                )
                row = {
                    "config": config_stem,
                    "resolution": resolution,
                    "seq_len": seq_len,
                    "batch_size": batch_size,
                    "latency_ms": latency_ms,
                    "fps": fps,
                    "peak_mem_mb": peak_mem_mb,
                    "encoder_mem_mb": encoder_mem_mb,
                    "oom": False,
                }
                print(
                    f"[{config_stem}] resolution={resolution} "
                    f"latency={latency_ms:.2f}ms fps={fps:.2f} "
                    f"peak_mem={peak_mem_mb:.1f}MB "
                    f"encoder_mem={encoder_mem_mb:.1f}MB",
                    flush=True,
                )
            except torch.cuda.OutOfMemoryError:
                row = {
                    "config": config_stem,
                    "resolution": resolution,
                    "seq_len": seq_len,
                    "batch_size": batch_size,
                    "latency_ms": None,
                    "fps": None,
                    "peak_mem_mb": None,
                    "encoder_mem_mb": None,
                    "oom": True,
                }
                print(
                    f"[OOM] {config_stem} OOMed at resolution {resolution}, "
                    f"skipping remaining resolutions",
                    flush=True,
                )
                writer.writerow(row)
                f.flush()
                wandb.log(
                    {
                        "resolution": resolution,
                        "seq_len": seq_len,
                        "latency_ms": float("nan"),
                        "fps": float("nan"),
                        "peak_mem_mb": float("nan"),
                        "encoder_mem_mb": float("nan"),
                        "oom": 1,
                    },
                    step=resolution,
                )
                torch.cuda.empty_cache()
                break

            writer.writerow(row)
            f.flush()
            wandb.log(
                {
                    "resolution": resolution,
                    "seq_len": seq_len,
                    "latency_ms": latency_ms,
                    "fps": fps,
                    "peak_mem_mb": peak_mem_mb,
                    "encoder_mem_mb": encoder_mem_mb,
                    "oom": 0,
                },
                step=resolution,
            )

    table = wandb.Table(columns=fieldnames)
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            table.add_data(*[row[k] for k in fieldnames])
    wandb.log({"benchmark_table": table})
    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "config",
        nargs="?",
        default=None,
        help="Config file to load (e.g. experiment_2.yaml)",
    )
    parser.add_argument(
        "--config",
        dest="config_flag",
        default=None,
        help="Config file to load (e.g. experiment_2.yaml)",
    )
    parser.add_argument(
        "--resolutions",
        nargs="+",
        type=int,
        default=DEFAULT_RESOLUTIONS,
        help="Square resolutions to sweep.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Batch size for the forward pass.",
    )
    args = parser.parse_args()
    config_file = args.config_flag or args.config or "experiment_2.yaml"
    main(
        config_file=config_file,
        resolutions=args.resolutions,
        batch_size=args.batch_size,
    )