---
kernelspec:
  name: python3
  display_name: Python 3
---

# Run

It is good to understand some of the details of the run loop.

```python
for event in feed.play(...):
    account = broker.sync(event)                # execute fills and update the account
    singals = strategy.create_signals(event)    # generate signals from market data
    orders = trader.create_orders(signals, ...) # apply risk rules, create orders
    broker.place_orders(orders)                 # place orders at broker
    journal.track(...)                          # record metrics (optional)
```

## Step-by-step

1. **`broker.sync(event)`** — Open orders from previous steps are tested against prices and executed if conditions are met. No look-ahead bias: orders placed at time `t` only execute at time `t+1`. Returns an updated `Account` object which reflects the latest market data.

    Orders and positions that are closed, are not included in the returned account object.   

2. **`strategy.create_signals(event)`** — The strategy examines the event's price data and returns a list of `Signal` objects. Each signal has an asset, a rating (typically -1.0 to 1.0), and a type (`ENTRY`, `EXIT`, or `ENTRY_EXIT`). Strategies are **pure decision-makers** — they know nothing about cash, positions, or risk.

3. **`trader.create_orders(signals, event, account)`** — The trader applies risk management rules (position sizing, shorting constraints, order limits) and converts signals into concrete `Order` objects. Unlike strategies, traders **have full access to the Account** (cash, positions, buying power).

4. **`broker.place_orders(orders)`** — New orders are submitted to the broker. In `SimBroker`, they are stored and only evaluated for execution when the next event arrives.

5. **`journal.track(...)`** — Optional logging and metrics collection. Journals are passive observers that never modify state.



## Basic Backtest
A simple back test that iterates over all the historic data in the feed,
just requires a few lines of code.

```{code-cell} python
import roboquant as rq

feed = rq.feeds.YahooFeed("JPM", "IBM", start_date="2015-01-01")
strategy = rq.strategies.EMACrossover()
account = rq.run(feed, strategy)
print(account)
```

This works because `run()` provides sensible defaults: `SimBroker` (USD 1M deposit, 0% slippage) and `FlexTrader` (conservative position sizing).

## Custom Backtest

```{code-cell} python
from roboquant import USD

feed = rq.feeds.YahooFeed("AAPL", "MSFT", start_date="2020-01-01")
strategy = rq.strategies.EMACrossover()
trader = rq.traders.FlexTrader(shorting=True)
broker = rq.brokers.SimBroker(deposit=500_000@USD)
journal = rq.journals.MetricsJournal()

account = rq.run(feed, strategy, trader=trader, broker=broker, journal=journal)
```


## Walk Forward
Walk-forward analysis is a backtesting technique that mimics real-world trading by splitting historical data into successive periods. 


```{code-cell} python
timeframes = feed.timeframe().split(5)

for timeframe in timeframes:
    strategy = rq.strategies.EMACrossover(13, 26)
    account = rq.run(feed, strategy, timeframe=timeframe)
    print(f"{timeframe.strftime('%Y-%m-%d')} equity={account.equity():.0f}")
```


## Multi-run
A Multi-run samples a number of random time frames and then runs a back test on each of them.

```{code-cell} python
timeframes = feed.timeframe().sample(100, "365 days")
equities = []

for timeframe in timeframes:
    strategy = rq.strategies.EMACrossover(13, 26)
    account = rq.run(feed, strategy, timeframe=timeframe)
    equities.append(account.equity()[USD])

print(f"min={min(equities):.0f}, max={max(equities):.0f}")
```

## Live and paper-trade run
A live or paper-trade run is only different from a back test in the implementation
of two of the components selected.

Instead of a the `SimBroker` a real broker is selected and instead of a historic
data feed, a live datafeed is used.

```{code} python
import os
from dotenv import load_dotenv
from roboquant.third_party.alpaca import AlpacaLiveFeed, AlpacaBroker

load_dotenv()

# Setup the real broker
api_key =  os.environ["ALPACA_API_KEY"]
secret_key = os.environ["ALPACA_SECRET"]
broker = AlpacaBroker(api_key, secret_key)

# Setup the live feed
alpaca_feed = AlpacaLiveFeed(api_key, secret_key, market="iex")
symbols = ["TSLA", "MSFT", "NVDA", "AMD", "AAPL"]
alpaca_feed.subscribe_trades(*symbols)

# Run a strategy
strategy = rq.strategies.EMACrossover(13, 26)
timeframe = rq.Timeframe.next("120 min")
account = rq.run(feed, strategy, broker=broker, timeframe=timeframe)
```



