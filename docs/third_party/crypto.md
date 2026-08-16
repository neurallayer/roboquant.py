---
kernelspec:
  name: python3
  display_name: Python 3
---

# Crypto

## Overview
The integration with crypto exchanges is done by using the [CCXT](https://github.com/ccxt/ccxt) package.
It provides both market data and broker functionality from these exchanges.

---
## Supported Exchanges
CCXT supports **100+** exchanges. Some popular ones include:

| Exchange   | Class              |
|------------|--------------------|
| Binance    | `ccxt.binance()`   |
| Coinbase   | `ccxt.coinbase()`  |
| Kraken     | `ccxt.kraken()`    |
| Bybit      | `ccxt.bybit()`     |
| OKX        | `ccxt.okx()`       |

```{code-cell} python
import ccxt

# List first 10 exchanges
print(ccxt.exchanges[:10])
```
---
## Market Data
CCXT provides access to historical OHLCV data, order books, and live ticker feeds
from hundreds of cryptocurrency exchanges.

The following example shows a back test over historical data gotten from Binance exchange.
This uses a public API, so no trading account with Binance is required for this example.

```{code-cell} python
import ccxt
import roboquant as rq
from roboquant.feeds.crypto import CryptoFeed
from roboquant.common.monetary import USDT

rq.set_light_style()

exchange = ccxt.binance()
# exchange = ccxt.kraken()  # or any other exchange supported by ccxt
feed = CryptoFeed(exchange, "BTC/USDT", "ETH/USDT", start_date="2020-01-01 00:00:00", interval="1d")

strategy = rq.strategies.EMACrossover()
trader = rq.traders.FlexTrader(size_fractions=4, max_order_pct=0.2, max_position_pct=0.5, shorting=True)
broker = rq.brokers.SimBroker(deposit=10_000@USDT)
account = rq.run(feed, strategy, trader=trader, broker=broker)
print(account)

for asset in feed.assets():
    feed.plot(asset, trades=account.trades)
```
---
## Broker Integration

For broker integration you typically need a trading account with the Exchange/Broker you want to use.


```{code} python
  from dotenv import load_env
  load_env()

  key = os.getenv("ALPACA_API_KEY")
  secret = os.getenv("ALPACA_SECRET")
  assert(key is not None and secret is not None), "ALPACA_API_KEY and ALPACA_SECRET must be set"
  alpaca = ccxt.alpaca({
      "apiKey": key,
      "secret": secret
  })
  alpaca.set_sandbox_mode(True)
  broker = CryptoBroker(alpaca)
  account = broker.sync()
```