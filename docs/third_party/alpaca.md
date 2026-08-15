---
kernelspec:
  name: python3
  display_name: Python 3
---

# Alpaca

## Overview
[Alpaca Markets](https://alpaca.markets/) provides a commission-free trading API for US equities and crypto.
*Roboquant* integrates with Alpaca for both **live/paper trading** (broker) and **real-time/historical market data** (feed).

## Prerequisites

Install the Alpaca dependencies alongside roboquant:

```bash
pip install roboquant[extra]
```

Set your API credentials as environment variables (or put them in a .env file that you don't put under version control):

```bash
export ALPACA_API_KEY="your-api-key"
export ALPACA_SECRET_KEY="your-secret-key"
```

---

## AlpacaFeed — Market Data

The `AlpacaHistoricStockFeed` fetches historical stock data from Alpaca's API. It supports stocks with configurable bar sizes and timeframes.

### Basic Usage

```{code-cell} python
import os
from dotenv import load_dotenv

from datetime import datetime, timedelta
from roboquant.feeds.alpaca import AlpacaHistoricStockFeed, AlpacaHistoricCryptoFeed

load_dotenv()

api_key = os.environ["ALPACA_API_KEY"]
secret_key = os.environ["ALPACA_SECRET"]


# Fetch 1-minute bars for the last 5 trading days
feed = AlpacaHistoricStockFeed(api_key, secret_key)
feed.retrieve_bars("F", "TSLA", "JPM", start="2026-03-01", end="2026-03-02")
print(feed)

feed = AlpacaHistoricCryptoFeed(api_key, secret_key)
feed.retrieve_trades("BTC/USDT", start="2026-04-01", end="2026-04-10")
print(feed)
```

---

## AlpacaBroker — Live & Paper Trading

The `AlpacaBroker` allows you to execute orders against an Alpaca live or paper trading account. 
It can be used standalone or passed into {cl}`run` for live or paper trading execution.


```{code-cell} python
from roboquant.brokers.alpaca import AlpacaBroker

broker = AlpacaBroker(
    api_key=api_key,
    secret_key=secret_key
)

account = broker.sync()
print(account)
```
