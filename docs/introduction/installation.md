---
kernelspec:
  name: python3
  display_name: Python 3
---


# Installation

## Introduction
Roboquant can be installed like most other Python packages. It is being published on PyPI and
below you'll find instructions for various installation methods and scenarios.

:::{tip}
If you don't want to install anything, you can try the included Jupyter Notebooks online: [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/neurallayer/roboquant.py/HEAD?urlpath=%2Fdoc%2Ftree%2Fdocs%2Fnotebooks)

Or if you have a Google account, you can run the Notebooks on Google Colaboratory, or "Colab" for short. The Colab environments have more compute resources available, making the execution snappier. You can also use Gemini if you want to.

[![Back Test](https://img.shields.io/badge/Backtest_Notebook-00A1FF?logo=google-colab&logoColor=white)](https://colab.research.google.com/github/neurallayer/roboquant.py/blob/main/docs/notebooks/backtest.ipynb)
[![Charts](https://img.shields.io/badge/Chart_Notebook-00A1FF?logo=google-colab&logoColor=white)](https://colab.research.google.com/github/neurallayer/roboquant.py/blob/main/docs/notebooks/charts.ipynb)
:::

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

The core installation keeps the dependency footprint small.
If you need specific functionality, install roboquant with one or more of the following optional dependencies:

| Optional   | Description                                            | Install command |
|------------|--------------------------------------------------------|-----------------|
| `ai`       | AI/ML strategies using PyTorch and Stable-Baselines3   | `pip install --upgrade "roboquant[ai]"` |
| `extra`    | IBKR, Alpaca and Crypto support                        | `pip install --upgrade "roboquant[extra]"` |


To install roboquant with **all** optional dependencies at once:

```bash
pip install --upgrade "roboquant[ai,extra]"
```

:::{note}
Part of the `ai` dependency is PyTorch which is a very large package. So it can take some time before
it is completely downloaded and installed. 
:::


## Install with uv

[uv](https://github.com/astral-sh/uv) is a fast Python package and project manager. It is also the package manager
used to develop *roboquant*. It is highly recommended to check it out if you didn't do so already.

If you are using `uv`, starting a new project from scratch is straightforward:

```bash
mkdir my-project
cd my-project
uv init
uv add roboquant
uv run python -c "import roboquant;roboquant.info()"
```

Or with all the extras:

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
