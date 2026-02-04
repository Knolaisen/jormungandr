# Jormundgandr: End-to-End Video Object Detection with Spatial-Temporal Mamba

<div align="center">

![GitHub Workflow Status (with event)](https://img.shields.io/github/actions/workflow/status/Knolaisen/jormundgandr/ci.yml)
![GitHub top language](https://img.shields.io/github/languages/top/Knolaisen/jormundgandr)
![GitHub language count](https://img.shields.io/github/languages/count/Knolaisen/jormundgandr)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Project Version](https://img.shields.io/badge/version-0.0.1-blue)](https://img.shields.io/badge/version-0.0.1-blue)

<img src="docs/images/project-logo.png" width="50%" alt="Jormundgandr VOD Logo" style="display: block; margin-left: auto; margin-right: auto;">
</div>

<details> 
<summary><b>📋 Table of contents </b></summary>

- [Jormundgandr: End-to-End Video Object Detection with Spatial-Temporal Mamba](#jormundgandr-end-to-end-video-object-detection-with-spatial-temporal-mamba)
  - [Description](#description)
  - [Prerequisites](#prerequisites)
  - [Getting started](#getting-started)
  - [Usage](#usage)
    - [📖 Generate Documentation Site](#-generate-documentation-site)
  - [Testing](#testing)
  - [Authors](#authors)
    - [License](#license)

</details>

## Description

Jormundgandr is an novel end-to-end video object detection system that leverages the Spatial-Temporal Mamba architecture to accurately detect and track objects across video frames. By combining spatial and temporal information, Jormundgandr enhances detection accuracy and robustness, making it suitable for various applications such as surveillance, autonomous driving, and video analytics.

## Prerequisites

- **Git**: Ensure that git is installed on your machine. [Download Git](https://git-scm.com/downloads)
- **Python 3.12**: Required for the project. [Download Python](https://www.python.org/downloads/)
- **UV**: Used for managing Python environments. [Install UV](https://docs.astral.sh/uv/getting-started/installation/)
- **Docker** (optional): For DevContainer development. [Download Docker](https://www.docker.com/products/docker-desktop)

## Getting started

<!-- TODO: In this Section you describe how to install this project in its intended environment.(i.e. how to get it to run)  
-->

1. **Clone the repository**:

   ```sh
   git clone https://github.com/Knolaisen/jormundgandr.git
   cd jormundgandr
   ```

1. **Install dependencies**:

   ```sh
   uv sync
   ```

<!--
1. **Configure environment variables**:
    This project uses environment variables for configuration. Copy the example environment file to create your own:
    ```sh
    cp .env.example .env
    ```
    Then edit the `.env` file to include your specific configuration settings.
-->

1. **Set up pre commit** (only for development):
   ```sh
   uv run pre-commit install
   ```

## Usage

To run the project, run the following command from the root directory of the project:

```bash

```

<!-- TODO: Instructions on how to run the project and use its features. -->

### 📖 Generate Documentation Site

To build and preview the documentation site locally:

```bash
uv run mkdocs build
uv run mkdocs serve
```

This will build the documentation and start a local server at [http://127.0.0.1:8000/](http://127.0.0.1:8000/) where you can browse the docs and API reference. Get the documentation according to the latest commit on main by viewing the `gh-pages` branch on GitHub: [https://Knolaisen.github.io/jormundgandr/](https://Knolaisen.github.io/jormundgandr/).

## Testing

To run the test suite, run the following command from the root directory of the project:

```bash
uv run pytest --doctest-modules --cov=src --cov-report=html
```

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
