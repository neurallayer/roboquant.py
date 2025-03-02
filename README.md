
# ![roboquant logo](https://github.com/neurallayer/roboquant.py/raw/main/docs/roboquant_header.png)

![PyPI - Version](https://img.shields.io/pypi/v/roboquant)
![PyPI - License](https://img.shields.io/pypi/l/roboquant)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/roboquant)
![PyPI - Status](https://img.shields.io/pypi/status/roboquant)
![GitHub Actions Workflow Status](https://img.shields.io/github/actions/workflow/status/neurallayer/roboquant.py/python-package.yml)
[![discord](https://img.shields.io/discord/954650958300856340?label=discord)](https://discord.com/channels/954650958300856340/954650958300856343)

Roboquant is an open-source algorithmic trading platform. It is flexible, user-friendly and completely free to use. It is designed for anyone serious about algo-trading. 

So whether you are a beginning retail trader or an established trading firm, roboquant can help you to develop robust and fully automated trading strategies. You can find out more at [roboquant.org](https://roboquant.org).

## Usage
The following code snippet shows the steps required to run a full back-test on a number of stocks.

```python
import roboquant as rq

feed = rq.feeds.YahooFeed("JPM", "IBM", "F", start_date="2000-01-01")
strategy = rq.strategies.EMACrossover()
account = rq.run(feed, strategy)
print(account)
```

## Install
Roboquant can be installed like most other Python packages, using pip or conda.
Make sure you have Python version 3.11 or higher installed.

```shell
python3 -m pip install --upgrade roboquant
```
If you don't want to install anything locally, you can try out roboquant in an online Jupyter environment like
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/neurallayer/roboquant-notebooks/blob/main/intro_roboquant.ipynb)

The core of roboquant limits the number of dependencies. 
But you can install roboquant including one or more of the optional dependencies if you require certain functionality:

```shell
# PyTorch based strategies using RNNStrategy
python3 -m pip install --upgrade "roboquant[torch]"

# Integration with Interactive Brokers using IBKRBroker
python3 -m pip install --upgrade "roboquant[ibkr]"

# Integration with Alpaca
python3 -m pip install --upgrade "roboquant[alpaca]"

# Install Avro feed dependencies
python3 -m pip install --upgrade "roboquant[avro]"

# Install Avro feed dependencies
python3 -m pip install --upgrade "roboquant[parquet]"
```

## Building from source
Roboquant.py uses `uv` as the main tool for handling package dependencies.


```shell
uv sync --all-extras --dev
```

You should now be in the virtual environment and ready to build/install roboquant:

```shell
uv build
uv pip install .
```

Some other useful commands:

```shell
# run the unit tests
uv run python -m unittest discover -s tests/unit 

# validate the code
uvx ruff check

# publish, only works if UV_PUBLISH_TOKEN is set
uv publish 
```

## License
Roboquant is made available under the Apache 2.0 license. You can read more about the Apache 2.0 license on this page: https://www.apache.org/licenses/LICENSE-2.0.html

## Disclaimer
Absolutely no warranty is implied with this product. Use at your own risk. I provide no guarantee that it will be profitable, or that it won't lose all your money very quickly or does not contain bugs.

All financial trading offers the possibility of loss. Leveraged trading, may result in you losing all your money, and still owing more. Back tested results are no guarantee of future performance. I can take no responsibility for any losses caused by live trading using roboquant. Use at your own risk. I am not registered or authorised by any financial regulator.

## Kotlin version
Next to this Python version of `roboquant`, there is also a Kotlin version available. Both share a similar API, just the used computer language is different. Which one to use depends very much on personal preferences, skills and usage.
