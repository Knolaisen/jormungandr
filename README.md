# Jormungandr: End-to-End Video Object Detection with Spatial-Temporal Mamba

<div align="center">

![GitHub Workflow Status (with event)](https://img.shields.io/github/actions/workflow/status/Knolaisen/jormungandr/ci.yml)
![GitHub top language](https://img.shields.io/github/languages/top/Knolaisen/jormungandr)
![GitHub language count](https://img.shields.io/github/languages/count/Knolaisen/jormungandr)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Project Version](https://img.shields.io/badge/version-0.0.1-blue)](https://img.shields.io/badge/version-0.0.1-blue)

<img src="docs/images/project-logo.png" width="50%" alt="jormungandr VOD Logo" style="display: block; margin-left: auto; margin-right: auto;">
</div>

<details> 
<summary><b>📋 Table of contents </b></summary>

- [Jormungandr: End-to-End Video Object Detection with Spatial-Temporal Mamba](#jormungandr-end-to-end-video-object-detection-with-spatial-temporal-mamba)
  - [Description](#description)
  - [Getting started](#getting-started)
    - [Installation](#installation)
  - [Usage](#usage)
    - [Still Image Detection (Fafnir)](#still-image-detection-fafnir)
    - [Video Object Detection (Jormungandr)](#video-object-detection-jormungandr)
    - [Pretrained Models](#pretrained-models)
  - [Documentation](#documentation)
  - [Authors](#authors)
    - [License](#license)

</details>

## Description

Jormungandr is an novel end-to-end video object detection system that leverages the Spatial-Temporal Mamba architecture to accurately detect and track objects across video frames. By combining spatial and temporal information, Jormungandr enhances detection accuracy and robustness, making it suitable for various applications such as surveillance, autonomous driving, and video analytics.

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
Use `Jormungandr` for end-to-end video object detection using spatial-temporal modeling.
```python
import torch
from jormungandr import Jormungandr

device = torch.device("cuda")

batch, frames, channels, height, width = 32, 8, 3, 224, 224
x = torch.randn(batch, frames, channels, height, width).to(device)

# Initialize model
model = Jormungandr(variant="jormungandr-b", pretrained=True).to(device)
model.eval()

# Inference
with torch.no_grad():
    detections = model(x)
```

### Pretrained Models

We provide pretrained models hosted on [Hugging Face](https://huggingface.co/SverreNystad).

* The **Fafnir** models (`fafnir-t`, `fafnir-s`, `fafnir-b`) are pretrained on the [COCO](https://cocodataset.org/#home) dataset.  
* The **Jormungandr** models (`jormungandr-t`, `jormungandr-s`, `jormungandr-b`) are pretrained on the [MOT17](https://motchallenge.net/data/MOT17/) dataset.

These models will be automatically downloaded when initialized in your code.

## Documentation

* [**Architecture Design**](docs/architectural_design.md)
* [**Developer Setup Guide**](docs/developer_setup.md)
* [**API Reference**](https://knolaisen.github.io/jormungandr/)




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
