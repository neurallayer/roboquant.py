---
kernelspec:
  name: python3
  display_name: Python 3
---

# MetaTrader

## Overview
MetaTrader (MT4/MT5) is a widely used trading platform for accessing broker-provided
market data and executing trades. 

Roboquant uses the TickerAll REST API to access this market data and trading
functionality for any MT4/MT5 compatible broker. 

## TickerAll API
Create an API key with TickerAll and store it in the `TICKERALL_API_KEY`
environment variable. You can do this at the [TickerAll website](https://tickerall.com)

You of course also need to already have a MT4 or MT5 broker.  

## Broker Example
The following example shows how to cancel orders and close positions using the
`TickerAllBroker`.

```{code} python
import os
from time import sleep
from dotenv import load_dotenv
import roboquant as rq
from roboquant.brokers.tickerall import TickerAllBroker

load_dotenv()

# TickerAll API key
KEY = os.environ["TICKERALL_API_KEY"]

# Your Broker details
MT5_SERVER = os.environ["MT5_SERVER"]
MT5_ACCOUNT = os.environ["MT5_ACCOUNT"]
MT5_PASSWORD = os.environ["MT5_PASSWORD"]

broker = TickerAllBroker.connect(
            api_key = KEY,
            broker= "mt5",
            server=MT5_SERVER,
            account=MT5_ACCOUNT,
            password=MT5_PASSWORD
          )

def place_orders(orders):
    broker.place_orders(orders)
    sleep(5)
    account = broker.sync()
    print(account, "\n", flush=True)
    return account

try:
    account = broker.sync()
    print(account, "\n", flush=True)

    i = input("removed orders y/n: ")
    if i == "y":
        orders = [order.cancel() for order in account.orders]
        account = place_orders(orders)

    i = input("close positions y/n: ")
    if i == "y":
        orders = [pos.close_order() for pos in account.positions]
        account = place_orders(orders)

finally:
  broker.close()

```

## Feed example
Below is an example how to use a live feed from your MetaTrader broker.

```{code} python
import roboquant as rq
from roboquant.feeds.tickerall import TickerAllLiveFeed

feed = TickerAllLiveFeed.connect(
    api_key = KEY,
    broker= "mt5",
    server=MT5_SERVER,
    account=MT5_ACCOUNT,
    password=MT5_PASSWORD
)

feed.subscribe("BTCUSD", "EURUSD")
tf = rq.Timeframe.next("10 min")
for event in feed.play(tf):
  print(event)
```

