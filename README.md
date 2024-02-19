
# Roboquant

![PyPI - Version](https://img.shields.io/pypi/v/roboquant)
![PyPI - License](https://img.shields.io/pypi/l/roboquant)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/roboquant)
![PyPI - Status](https://img.shields.io/pypi/status/roboquant)

Roboquant is an open-source algorithmic trading platform. It is flexible, user-friendly and completely free to use. It is designed for anyone serious about algo-trading. So whether you are a beginning retail trader or an established trading firm, roboquant can help you to develop robust and fully automated trading strategies.

![roboquant logo](https://github.com/neurallayer/roboquant.py/raw/main/docs/roboquant_header.png)

## Usage
The following code snippet shows the steps required to run a full back-test on a number of stocks.

```python
from roboquant import *

feed = YahooFeed("TSLA", "AMZN", "IBM")
strategy = EMACrossover()
roboquant = Roboquant(strategy)
tracker = StandardTracker()

roboquant.run(feed, tracker)
print(tracker)
```

## Install
Roboquant can be installed like most other Python packages, using for example pip or conda. Just make sure you have Python version 3.10 or higher installed.

```shell
python3 -m pip install --upgrade roboquant
```

If you want to use YahooFinance market data or PyTorch based strategies, you can install roboquant including one or more of the optional dependencies:

```shell
python3 -m pip install --upgrade roboquant[yahoo]
python3 -m pip install --upgrade roboquant[torch]
python3 -m pip install --upgrade roboquant[yahoo,torch]
```

## Building from source
Although this first step isn't required, it is recommended to create a virtual environment. Go to directory where you have downloaded the source code and run the following commands:

```shell
python3 -m venv .venv
source .venv/bin/activate
```

You should now be in the virtual environment. Ready to install the required packages and build roboquant:

```shell
pip install -r requirements.txt
python -m build
```

Some other useful commands:

```shell
# run unittests
python -m unittest discover -s tests/unit 

# validate code
flake8 roboquant tests

# install locally
pip install .
```


## Interactive Brokers
Unfortunatly Interactive Brokers doesn't allow their Python client library to be redistributed by third parties. However it is freely available to be downloaded and installed. Please follow the instructions found [here](https://ibkrcampus.com/ibkr-quant-news/interactive-brokers-python-api-native-a-step-by-step-guide/) (download and install version 10.19).  

# Kotlin version
Next to this Python version of `roboquant`, there is also a Koltin version available. Both (will) share a similar API, just the used computer language is different.

Which one to use, depends very much on personal preference, skills and use-case.
