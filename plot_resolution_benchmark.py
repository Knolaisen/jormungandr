"""
Render the two-panel resolution-benchmark figure from per-config CSVs.

Reads each CSV produced by benchmark_resolution.py and draws side-by-side
panels of FPS vs resolution and peak GPU memory vs resolution. The cliff at
which a model OOMed is marked at the last fitted resolution.
"""

import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def _str_to_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes"}


def plot(
    csv_paths: list[Path],
    labels: list[str],
    output: Path,
    batch_size: int,
) -> plt.Figure:
    assert len(csv_paths) == len(labels), "Got %d csvs and %d labels" % (
        len(csv_paths),
        len(labels),
    )

    fig, (ax_fps, ax_mem) = plt.subplots(1, 2, figsize=(11, 4.5))
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for i, (csv_path, label) in enumerate(zip(csv_paths, labels)):
        color = colors[i % len(colors)]
        df = pd.read_csv(csv_path)
        df["oom"] = df["oom"].apply(_str_to_bool)
        fitted = df[~df["oom"]].sort_values("resolution")
        ooms = df[df["oom"]].sort_values("resolution")

        ax_fps.plot(
            fitted["resolution"], fitted["fps"],
            marker="o", color=color, label=label,
        )
        ax_mem.plot(
            fitted["resolution"], fitted["peak_mem_mb"] / 1024.0,
            marker="o", color=color, label=label,
        )

        if not ooms.empty and not fitted.empty:
            cliff_res = int(ooms["resolution"].iloc[0])
            last_x = int(fitted["resolution"].iloc[-1])
            last_fps = float(fitted["fps"].iloc[-1])
            last_mem = float(fitted["peak_mem_mb"].iloc[-1]) / 1024.0
            ax_fps.annotate(
                f"OOM @ {cliff_res}",
                xy=(last_x, last_fps),
                xytext=(8, 8), textcoords="offset points",
                color=color, fontsize=9,
                arrowprops=dict(arrowstyle="->", color=color, lw=1),
            )
            ax_mem.annotate(
                f"OOM @ {cliff_res}",
                xy=(last_x, last_mem),
                xytext=(8, -16), textcoords="offset points",
                color=color, fontsize=9,
                arrowprops=dict(arrowstyle="->", color=color, lw=1),
            )

    ax_fps.set_xscale("log", base=2)
    ax_mem.set_xscale("log", base=2)
    ax_fps.set_xlabel("Image resolution (R, square R×R)")
    ax_mem.set_xlabel("Image resolution (R, square R×R)")
    ax_fps.set_ylabel("Throughput (images / s)")
    ax_mem.set_ylabel("Peak GPU memory (GB)")
    ax_fps.set_title("Inference speed")
    ax_mem.set_title("GPU memory")
    for ax in (ax_fps, ax_mem):
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()

    fig.suptitle(
        f"Speed and GPU memory vs image resolution (batch={batch_size}, A100 80GB)"
    )
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150)
    print(f"Saved figure to {output}")
    return fig


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csvs",
        nargs="+",
        required=True,
        type=Path,
        help="One or more CSVs produced by benchmark_resolution.py.",
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        required=True,
        help="Legend label for each CSV (same order).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/images/resolution_benchmark.png"),
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Skip logging the figure to W&B even if credentials are present.",
    )
    args = parser.parse_args()

    fig = plot(args.csvs, args.labels, args.output, args.batch_size)

    if not args.no_wandb and os.getenv("WANDB_API_KEY"):
        import wandb
        from jormungandr.config.configuration import (
            WANDB_API_KEY,
            WANDB_ENTITY,
            WANDB_PROJECT,
        )
        wandb.login(key=WANDB_API_KEY)
        wandb.init(
            project=WANDB_PROJECT,
            entity=WANDB_ENTITY,
            name="benchmark_resolution_plot",
            tags=["benchmark", "resolution", "plot"],
        )
        wandb.log({"resolution_benchmark": wandb.Image(fig)})
        wandb.finish()
    plt.close(fig)


if __name__ == "__main__":
    main()
