# Jormungandr: End-to-End Video Object Detection with Spatial-Temporal Mamba

<div align="center">

![GitHub Workflow Status (with event)](https://img.shields.io/github/actions/workflow/status/Knolaisen/jormungandr/ci.yml)
![GitHub top language](https://img.shields.io/github/languages/top/Knolaisen/jormungandr)
![GitHub language count](https://img.shields.io/github/languages/count/Knolaisen/jormungandr)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
![PyPI - Version](https://img.shields.io/pypi/v/jormungandr-ssm)


<img src="https://raw.githubusercontent.com/Knolaisen/jormungandr/refs/heads/main/docs/images/project-logo.png" width="50%" alt="jormungandr VOD Logo" style="display: block; margin-left: auto; margin-right: auto;">
</div>

<details> 
<summary><b>📋 Table of contents </b></summary>

- [Jormungandr: End-to-End Video Object Detection with Spatial-Temporal Mamba](#jormungandr-end-to-end-video-object-detection-with-spatial-temporal-mamba)
  - [Description](#description)
  - [Getting started](#getting-started)
    - [Prerequisites](#prerequisites)
    - [Installation](#installation)
  - [Usage](#usage)
    - [Still Image Detection (Fafnir)](#still-image-detection-fafnir)
    - [Video Object Detection (Jormungandr)](#video-object-detection-jormungandr)
    - [Pretrained Models](#pretrained-models)
    - [Visualizing the Loss Landscape](#visualizing-the-loss-landscape)
  - [Documentation](#documentation)
  - [Authors](#authors)
    - [License](#license)

</details>

## Description

Jormungandr is an novel end-to-end video object detection system that leverages the Spatial-Temporal Mamba architecture to accurately detect and track objects across video frames. By combining spatial and temporal information, Jormungandr enhances detection accuracy and robustness, making it suitable for various applications such as collision avoidance, search and rescue operations, surveillance, autonomous driving, and video analytics.

## Getting started

### Prerequisites

Before installing this package, ensure that your system meets the following requirements:

- **Operating System:** Linux
- **Python:** Version 3.12 or higher
- **Hardware:** CUDA-enabled GPU
- **Software Dependencies:**
  - NVIDIA drivers compatible with your GPU
  - CUDA Toolkit properly installed and configured, can be checked with `nvidia-smi`

### Installation

PyPI package:

```bash
pip install jormungandr-ssm
```

Alternatively, from source:

```bash
pip install git+https://github.com/Knolaisen/jormungandr
```

## Usage

We expose several levels of interface with the **Fafnir** still image detector and **Jormungandr** Video Object Detection (VOD) model. Both models follow a simple PyTorch-style API. Due to the Mamba architecture, the models are optimized for GPU execution and require CUDA for inference and training.

### Still Image Detection (Fafnir)


<img src="https://raw.githubusercontent.com/Knolaisen/jormungandr/refs/heads/main/docs/images/Fafnir.png" width="90%" alt="Fafnir architecture" style="display: block; margin-left: auto; margin-right: auto;">
</div>


Use `Fafnir` when performing object detection on single images.

```python
import torch
from jormungandr import Fafnir

device = torch.device("cuda")

batch, channels, height, width = 2, 3, 224, 224
x = torch.randn(batch, channels, height, width).to(device)

# Initialize model
model = Fafnir(variant="fafnir-b", pretrained=True).to(device)
model.eval()

# Inference
with torch.no_grad():
    detections = model(x)
```

### Video Object Detection (Jormungandr)

<img src="https://raw.githubusercontent.com/Knolaisen/jormungandr/refs/heads/main/docs/images/Jormungandr.png" width="90%" alt="Jormungandr architecture" style="display: block; margin-left: auto; margin-right: auto;">
</div>

Use `Jormungandr` for end-to-end video object detection using spatial-temporal modeling.

```python
import torch
from jormungandr import Jormungandr

device = torch.device("cuda")

frames, channels, height, width = 8, 3, 224, 224
x = torch.randn(frames, channels, height, width).to(device)

# Initialize model
model = Jormungandr(variant="jormungandr-b", pretrained=True).to(device)
model.eval()

# Inference
with torch.no_grad():
    detections = model(x)
```

### Pretrained Models

We provide pretrained models hosted on [Hugging Face](https://huggingface.co/SverreNystad).

- The **Fafnir** models (`fafnir-t`, `fafnir-s`, `fafnir-b`) are pretrained on the [COCO](https://cocodataset.org/#home) dataset.
- The **Jormungandr** models (`jormungandr-t`, `jormungandr-s`, `jormungandr-b`) are pretrained on the [MOT17](https://motchallenge.net/data/MOT17/) dataset.

These models will be automatically downloaded when initialized in your code.

### Visualizing the Loss Landscape

`scripts/plot_loss_landscape.py` plots the loss landscape of a trained **Fafnir** checkpoint on COCO using the filter-normalized random-direction method from [Li et al. 2018](https://arxiv.org/abs/1712.09913). It loads a W&B model artifact, samples two filter-normalized directions in parameter space, and evaluates the training loss on a 2-D grid around the checkpoint.

**Requirements:** `WANDB_API_KEY`, `WANDB_PROJECT`, and `WANDB_ENTITY` set in `.env` (same as for `train.py`); a CUDA-enabled GPU; a checkpoint already uploaded as a W&B model artifact.

**Smoke test** (under a minute; sanity-checks the wiring):

```bash
python scripts/plot_loss_landscape.py \
    --config-name <yaml-that-matches-the-checkpoint>.yaml \
    --wandb-artifact <entity>/<project>/<name>:<version> \
    --grid-size 5 --subset-size 16 --num-eval-batches 1 \
    --no-wandb-log
```

The printed `baseline loss at theta*` should match the validation loss the checkpoint was trained to, and the centre cell of the resulting grid should equal that baseline within numerical noise.

**Full run** (defaults: 41x41 grid over `[-1, 1]²`, 256 COCO val images, batch size 4):

```bash
python scripts/plot_loss_landscape.py \
    --config-name <yaml-that-matches-the-checkpoint>.yaml \
    --wandb-artifact <entity>/<project>/<name>:<version>
```

Outputs land in `plots/`:

- `fafnir_landscape_<timestamp>.png` — side-by-side contour and 3-D surface.
- `fafnir_landscape_<timestamp>.npz` — `alphas`, `betas`, `loss_grid`, plus run metadata, so you can re-plot without re-running.

Unless `--no-wandb-log` is passed, the figure is also logged to a W&B run with `job_type=loss_landscape`. Flags worth knowing:

| Flag | Default | Notes |
| --- | --- | --- |
| `--grid-size N` | `41` | Sweeps `N x N` cells. 41 → 1681 forward-pass passes; expect 20–60 min on one GPU. |
| `--extent E` | `1.0` | `alpha, beta` range is `[-E, E]`. |
| `--subset-size K` | `256` | COCO val images used at each grid cell. |
| `--batch-size B` | `4` | Eval batch size. |
| `--num-eval-batches M` | all | Cap on batches per cell. Lower this for a faster (noisier) plot. |
| `--seed S` | `0` | Seed for the two random directions. |
| `--output-dir DIR` | `plots` | Where to write the `.png` and `.npz`. |
| `--no-wandb-log` | off | Skip the final `wandb.init`/`wandb.log` call. |

The script reproduces the trainer's end-of-epoch unfreeze state before sampling directions, so the landscape reflects exactly the parameter subspace the optimizer was moving in at the end of training. The video model (`Jormungandr`) is not yet supported — the same recipe extends to it but needs to handle the spatial/temporal encoder pair.

## Documentation

- [**Architecture Design**](docs/architectural_design.md)
- [**Developer Setup Guide**](docs/developer_setup.md)
- [**API Reference**](https://knolaisen.github.io/jormungandr/)

## Authors

<table align="center">
    <tr>
      <td align="center">
        <a href="https://github.com/Knolaisen">
          <img src="https://github.com/Knolaisen.png?size=100" width="100px;" alt="Kristoffer Nohr Olaisen"/><br />
          <sub><b>Kristoffer Nohr Olaisen</b></sub>
        </a>
      </td>
      <td align="center">
        <a href="https://github.com/SverreNystad">
          <img src="https://github.com/SverreNystad.png?size=100" width="100px;" alt="Sverre Nystad"/><br />
          <sub><b>Sverre Nystad</b></sub>
        </a>
      </td>
    </tr>
</table>

### License

______________________________________________________________________

Distributed under the MIT License. See `LICENSE` for more information.
