---
kernelspec:
  name: python3
  display_name: Python 3
---


# Installation

Roboquant can be installed like most other Python packages. Below you'll find instructions for various installation methods and scenarios.

## Prerequisites

- **Python 3.12 or higher** — roboquant uses modern Python features and does not support older versions.
- A **virtual environment** is recommended to keep dependencies isolated.

## Install with pip

The simplest way to install roboquant is via `pip`:

```bash
pip install --upgrade roboquant
```

This installs the **core** package with a minimal set of dependencies.

## Optional Dependencies

The core installation keeps the dependency footprint small. If you need specific functionality, install roboquant with one or more of the following extras:

| Optional   | Description                                            | Install command |
|------------|--------------------------------------------------------|-----------------|
| `ai`       | AI/ML strategies using PyTorch and Stable-Baselines3   | `pip install --upgrade "roboquant[ai]"` |
| `extra`    | IBKR, Alpaca and Crypto support                        | `pip install --upgrade "roboquant[extra]"` |


To install roboquant with **all** optional dependencies at once:

```bash
pip install --upgrade "roboquant[ai,extra]"
```

## Install with uv

[uv](https://github.com/astral-sh/uv) is a fast Python package and project manager. If you are using `uv`, getting started is straightforward:

```bash
mkdir my-project
cd my-project
uv init
uv add roboquant
uv run python -c "import roboquant;roboquant.info()"
```

With extras:

```bash
uv add "roboquant[ai,extra]"
```

## Install from Source

To install the latest development version directly from the GitHub repository:

```bash
pip install git+https://github.com/neurallayer/roboquant.py.git
```

Or clone the repository and install locally. This assumes you already have `uv` installed since that 
is used for the build process.

```bash
git clone https://github.com/neurallayer/roboquant.py.git
cd roboquant.py
./bin/local_install.sh
```

## Verify Installation

To confirm that roboquant was installed correctly, run the following in a Python shell:

```{code-cell} python
import roboquant as rq
rq.info()
```

You should see the installed version number printed.

You can also run a quick smoke test:

```{code-cell} python
import roboquant as rq
account = rq.demo_run()
print(account)
```

If this runs without errors, your installation is ready to use.
