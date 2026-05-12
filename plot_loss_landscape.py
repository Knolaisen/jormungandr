"""
Plot the Fafnir loss landscape on COCO.

Implements the Li et al. 2018 method ("Visualizing the Loss Landscape of
Neural Nets"): pick two filter-normalized random directions in parameter
space and evaluate the trained loss on a 2-D grid around the checkpoint.

Usage:
    python scripts/plot_loss_landscape.py \
        --config-name experiment_2_ffn.yaml \
        --wandb-artifact entity/project/name:v0 \
        [--grid-size 41] [--extent 1.0] \
        [--subset-size 256] [--batch-size 4] \
        [--num-eval-batches N] [--seed 0]
"""

from __future__ import annotations

import argparse
import os
import shutil
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb
from tqdm import tqdm

from jormungandr.config.configuration import (
    Config,
    FafnirConfig,
    WANDB_API_KEY,
    WANDB_ENTITY,
    WANDB_PROJECT,
    load_config,
)
from jormungandr.datasets import create_dataloaders
from jormungandr.fafnir import Fafnir
from jormungandr.training.criterion import build_criterion


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config-name", default="config.yaml",
                   help="YAML in src/jormungandr/config/ matching the checkpoint.")
    p.add_argument("--wandb-artifact", required=True,
                   help="W&B artifact, e.g. 'entity/project/mamba2-giou:latest'.")
    p.add_argument("--grid-size", type=int, default=41,
                   help="Grid resolution N; sweeps N x N cells.")
    p.add_argument("--extent", type=float, default=1.0,
                   help="alpha,beta range is [-extent, extent].")
    p.add_argument("--subset-size", type=int, default=256,
                   help="COCO val subset size.")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--num-eval-batches", type=int, default=None,
                   help="Cap batches per grid cell. Default: use all available.")
    p.add_argument("--seed", type=int, default=0,
                   help="Seed for the two random directions.")
    p.add_argument("--output-dir", default="plots")
    p.add_argument("--device", default="cuda")
    p.add_argument("--no-wandb-log", action="store_true",
                   help="Skip logging the final figure as a W&B run.")
    return p.parse_args()


def download_checkpoint(artifact_name: str) -> tuple[str, str]:
    api = wandb.Api()
    artifact = api.artifact(artifact_name, type="model")
    art_dir = artifact.download()
    files = [f for f in os.listdir(art_dir) if not f.startswith(".")]
    if not files:
        raise RuntimeError(f"No files inside artifact dir {art_dir}")
    return os.path.join(art_dir, files[0]), art_dir


def apply_final_unfreezing(model: Fafnir, config: Config) -> None:
    """Mirror `_handle_unfreezing` for the last training epoch.

    Why: backbone/decoder/head may be frozen at start of training and unfrozen
    later. To match the parameter subspace the optimizer actually moves in
    *at the end* of training, replay the final-epoch unfreeze state.
    """
    final_epoch = max(0, config.trainer.epochs - 1)
    if config.model.backbone.freeze_backbone:
        unfreeze = final_epoch >= config.trainer.epoch_to_unfreeze_backbone
        for p in model.backbone.parameters():
            p.requires_grad = unfreeze
    if config.model.decoder.freeze_decoder:
        unfreeze = final_epoch >= config.trainer.epoch_to_unfreeze_decoder
        for p in model.decoder.parameters():
            p.requires_grad = unfreeze
    if config.model.output_head.freeze_prediction_head:
        unfreeze = final_epoch >= config.trainer.epoch_to_unfreeze_output_head
        for p in model.output_head.parameters():
            p.requires_grad = unfreeze


def snapshot_trainable_params(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}


def make_filter_normalized_direction(
    theta_star: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Sample d ~ N(0, I) per tensor, then filter-normalize.

    For tensors with >= 2 dims, rescale each slice along dim 0 so its L2 norm
    matches the corresponding slice of theta_star. For 1-D tensors (biases,
    LayerNorm gamma/beta, Mamba A_log/D), set d = 0 -- the Li et al.
    convention. Otherwise these tiny-shape params dominate the perturbation.
    """
    direction: dict[str, torch.Tensor] = {}
    for name, theta in theta_star.items():
        if theta.dim() < 2:
            direction[name] = torch.zeros_like(theta)
            continue
        d = torch.randn_like(theta)
        flat_d = d.reshape(theta.size(0), -1)
        flat_theta = theta.reshape(theta.size(0), -1)
        d_norms = flat_d.norm(dim=1, keepdim=True).clamp_min(1e-10)
        t_norms = flat_theta.norm(dim=1, keepdim=True)
        scale = (t_norms / d_norms).reshape(-1, *([1] * (theta.dim() - 1)))
        direction[name] = d * scale
    return direction


def set_perturbed_weights(
    model: torch.nn.Module,
    theta_star: dict[str, torch.Tensor],
    d1: dict[str, torch.Tensor],
    d2: dict[str, torch.Tensor],
    alpha: float,
    beta: float,
) -> None:
    with torch.no_grad():
        for n, p in model.named_parameters():
            if n in theta_star:
                p.copy_(theta_star[n] + alpha * d1[n] + beta * d2[n])


def restore_weights(model: torch.nn.Module, theta_star: dict[str, torch.Tensor]) -> None:
    with torch.no_grad():
        for n, p in model.named_parameters():
            if n in theta_star:
                p.copy_(theta_star[n])


@torch.no_grad()
def evaluate_loss(
    model: Fafnir,
    batches: list[dict],
    criterion,
    config: Config,
    device: torch.device | str,
) -> float:
    total, n = 0.0, 0
    for data in batches:
        pixel_values = data["pixel_values"].to(device, non_blocking=True)
        pixel_mask = data["pixel_mask"].to(device, non_blocking=True)
        labels = [
            {k: v.to(device, non_blocking=True) for k, v in label.items()}
            for label in data["labels"]
        ]
        class_labels, bbox_coordinates, intermediate = model.forward(pixel_values, pixel_mask)
        output_class, output_coord = None, None
        if config.trainer.loss.auxiliary_loss:
            output_class, output_coord = model.output_head.forward(intermediate)
        loss, _, _ = criterion(
            logits=class_labels,
            labels=labels,
            device=device,
            pred_boxes=bbox_coordinates,
            config=config.trainer.loss,
            outputs_class=output_class,
            outputs_coord=output_coord,
        )
        total += loss.item()
        n += 1
    return total / max(n, 1)


def plot_landscape(alphas: np.ndarray, betas: np.ndarray, loss_grid: np.ndarray,
                   out_png_path: Path, title_suffix: str = "") -> None:
    fig = plt.figure(figsize=(14, 6))
    A, B = np.meshgrid(alphas, betas, indexing="ij")

    ax1 = fig.add_subplot(1, 2, 1)
    cs = ax1.contourf(A, B, loss_grid, levels=30, cmap="viridis")
    fig.colorbar(cs, ax=ax1, label="loss")
    ax1.scatter([0], [0], color="red", marker="x", s=80, label=r"$\theta^*$")
    ax1.set_xlabel(r"$\alpha$ (direction 1)")
    ax1.set_ylabel(r"$\beta$ (direction 2)")
    ax1.set_title(f"Fafnir loss landscape (contour) {title_suffix}".strip())
    ax1.legend()

    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    ax2.plot_surface(A, B, loss_grid, cmap="viridis", linewidth=0, antialiased=True)
    ax2.set_xlabel(r"$\alpha$")
    ax2.set_ylabel(r"$\beta$")
    ax2.set_zlabel("loss")
    ax2.set_title("3D surface")

    fig.tight_layout()
    fig.savefig(out_png_path, dpi=150)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    config = load_config(args.config_name)
    if not isinstance(config.model, FafnirConfig):
        raise SystemExit(
            f"This script only supports Fafnir; got {type(config.model).__name__}."
        )

    print(f"[1/6] Building Fafnir from {args.config_name}")
    model = Fafnir(config=config.model).to(device)

    print(f"[2/6] Downloading checkpoint {args.wandb_artifact}")
    wandb.login(key=WANDB_API_KEY)
    ckpt_path, art_dir = download_checkpoint(args.wandb_artifact)
    state_dict = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state_dict)
    shutil.rmtree(art_dir, ignore_errors=True)

    print("[3/6] Applying end-of-training unfreeze state")
    apply_final_unfreezing(model, config)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"    trainable params: {trainable:,} / {total:,}")

    print(f"[4/6] Building COCO val loader (subset_size={args.subset_size})")
    _, val_loader = create_dataloaders(
        dataset_identifier=config.trainer.dataset_name,
        batch_size=args.batch_size,
        seed=config.trainer.seed,
        shuffle=False,
        subset_size=args.subset_size,
    )
    criterion = build_criterion(config.trainer.loss.name)

    print("    materializing fixed eval batches")
    batches: list[dict] = []
    for i, data in enumerate(val_loader):
        if args.num_eval_batches is not None and i >= args.num_eval_batches:
            break
        batches.append(data)
    if not batches:
        raise SystemExit("No batches produced -- check --subset-size and --batch-size.")
    print(f"    using {len(batches)} batches per grid cell")

    model.eval()
    print(f"[5/6] Generating filter-normalized directions (seed={args.seed})")
    theta_star = snapshot_trainable_params(model)
    torch.manual_seed(args.seed)
    d1 = make_filter_normalized_direction(theta_star)
    d2 = make_filter_normalized_direction(theta_star)

    baseline = evaluate_loss(model, batches, criterion, config, device)
    print(f"    baseline loss at theta*: {baseline:.4f}")

    print(f"[6/6] Sweeping {args.grid_size}x{args.grid_size} grid over "
          f"[-{args.extent}, {args.extent}]")
    alphas = np.linspace(-args.extent, args.extent, args.grid_size)
    betas = np.linspace(-args.extent, args.extent, args.grid_size)
    loss_grid = np.full((args.grid_size, args.grid_size), np.nan, dtype=np.float32)

    total_cells = args.grid_size * args.grid_size
    with tqdm(total=total_cells, desc="grid", unit="cell") as pbar:
        for i, a in enumerate(alphas):
            for j, b in enumerate(betas):
                set_perturbed_weights(model, theta_star, d1, d2, float(a), float(b))
                loss_grid[i, j] = evaluate_loss(model, batches, criterion, config, device)
                pbar.update(1)
                pbar.set_postfix(a=f"{a:+.2f}", b=f"{b:+.2f}", loss=f"{loss_grid[i, j]:.3f}")
    restore_weights(model, theta_star)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    npz_path = out_dir / f"fafnir_landscape_{timestamp}.npz"
    png_path = out_dir / f"fafnir_landscape_{timestamp}.png"
    np.savez(
        npz_path,
        alphas=alphas, betas=betas, loss_grid=loss_grid,
        baseline=np.array(baseline),
        artifact=np.array(args.wandb_artifact),
        config_name=np.array(args.config_name),
        seed=np.array(args.seed),
        extent=np.array(args.extent),
        num_eval_batches=np.array(len(batches)),
    )
    plot_landscape(alphas, betas, loss_grid, png_path,
                   title_suffix=f"({config.trainer.loss.name})")
    print(f"Saved {npz_path}")
    print(f"Saved {png_path}")

    if not args.no_wandb_log:
        wandb.init(
            project=WANDB_PROJECT,
            entity=WANDB_ENTITY,
            job_type="loss_landscape",
            config={
                "artifact": args.wandb_artifact,
                "config_name": args.config_name,
                "grid_size": args.grid_size,
                "extent": args.extent,
                "subset_size": args.subset_size,
                "num_eval_batches": len(batches),
                "seed": args.seed,
                "baseline_loss": baseline,
            },
        )
        wandb.log({"loss_landscape": wandb.Image(str(png_path))})
        wandb.finish()


if __name__ == "__main__":
    main()
