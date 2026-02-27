# Developer Setup

## Prerequisites

Before installing this package, ensure that your system meets the following requirements:

- **Operating System:** Linux
- **Hardware:** CUDA-enabled GPU
- **Software Dependencies:**
  - NVIDIA drivers compatible with your GPU
  - CUDA Toolkit properly installed and configured, can be checked with `nvidia-smi`
  - **Git**: Ensure that git is installed on your machine. [Download Git](https://git-scm.com/downloads)
  - **Python 3.12**: Required for the project. [Download Python](https://www.python.org/downloads/)
  - **UV**: Used for managing Python environments. [Install UV](https://docs.astral.sh/uv/getting-started/installation/)
  - **Docker** (optional): For DevContainer development. [Download Docker](https://www.docker.com/products/docker-desktop)

### Development Installation

1. **Clone the repository**:

   ```sh
   git clone https://github.com/Knolaisen/jormungandr.git
   cd jormungandr
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

### 📖 Generate Documentation Site

To build and preview the documentation site locally:

```bash
uv run mkdocs build
uv run mkdocs serve
```

This will build the documentation and start a local server at [http://127.0.0.1:8000/](http://127.0.0.1:8000/) where you can browse the docs and API reference. Get the documentation according to the latest commit on main by viewing the `gh-pages` branch on GitHub: [https://Knolaisen.github.io/jormungandr/](https://Knolaisen.github.io/jormungandr/).

## Testing

To run the test suite, run the following command from the root directory of the project:

```bash
uv run pytest --doctest-modules --cov=src --cov-report=html
```
