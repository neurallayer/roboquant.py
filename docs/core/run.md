---
kernelspec:
  name: python3
  display_name: Python 3
---

# Run
(run_def)=
The `run()` function implements the main event loop in *roboquant*. It is used for everything
from back testing to live trading.

It is helpful to understand some of the details of the run loop.

```python
for event in feed.play(...):
    account = broker.sync(event)                # syncs and updates the account
    singals = strategy.create_signals(event)    # generate signals from market data
    orders = trader.create_orders(signals, ...) # apply risk rules, create orders
    broker.place_orders(orders)                 # place orders at broker
    journal.track(...)                          # record metrics (optional)
```

## Step-by-step

1. **`broker.sync(event)`** — Sync with the state of the underlying broker. Returns an updated {cl}`Account` object which reflects the latest state and market data. Orders and positions that are closed, are not included in the returned account object.   

2. **`strategy.create_signals(event)`** — The strategy examines the event's price data and returns a list of {cl}`Signal` objects. Each signal has an asset, a rating (typically -1.0 to 1.0), and a type (`ENTRY`, `EXIT`, or `ENTRY_EXIT`). Strategies are **pure decision-makers** — they know nothing about cash, positions, or risk.

3. **`trader.create_orders(signals, event, account)`** — The trader applies risk management rules (position sizing, shorting constraints, order limits) and converts signals into concrete `Order` objects. Unlike strategies, traders **have full access to the Account** (cash, positions, buying power).

4. **`broker.place_orders(orders)`** — New orders are submitted to the broker. In `SimBroker`, they are stored and only evaluated for execution when the next event arrives.

5. **`journal.track(...)`** — Optional logging and metrics collection. Journals are passive observers that never modify state.

:::{tip}
Any exception thrown during the execution of the run loop, will stop the run. However if you call the `stop_run()` function, the run is stopped early while still
regulary returning the latest account object.
:::

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

This works because `run()` provides sensible defaults: `SimBroker` (USD 1M deposit, 0% slippage) and `SimpleTrader`.

## Custom Backtest
The following snippets shows how to override many of the default settings. 

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
Below is very simple example of a walk forward that provides insights into the performance in different timeframes.

```{code-cell} python
timeframes = feed.timeframe().split(5)

for timeframe in timeframes:
    strategy = rq.strategies.EMACrossover(13, 26)
    account = rq.run(feed, strategy, timeframe=timeframe)
    print(f"{timeframe.strftime('%Y-%m-%d')} equity={account.equity():.0f}")
```

Often a walk forward is used in combination with hyper-parameter tuning.
This is known as **Walk-Forward Optimization (WFO)**. The idea is:

1. Split the historical data into a sequence of time windows (e.g. 5 periods).
2. For each window, use the *current* window as the **in-sample (training)** period to find the best parameters.
3. Test those parameters on the *next* window, the **out-of-sample (testing)** period.
4. Move forward one window and repeat.

This approach helps detect **overfitting**: if a parameter set performs well in-sample but poorly out-of-sample,
the strategy likely doesn't generalise. By measuring performance only on unseen data, you get a more
realistic estimate of how the strategy would have performed in production.

:::{note}
A common pitfall is peeking into the future — ensure the training window always ends before the testing window begins.
The example below respects this by using `timeframes[idx]` for training and `timeframes[idx+1]` for testing.
:::

In practice you might also track metrics across all out-of-sample periods (e.g. average Sharpe ratio, win rate)
rather than just the final equity value, giving a fuller picture of robustness.


```{code-cell} python
from collections import namedtuple
Best = namedtuple("Best", "equity param")

timeframes = feed.timeframe().split(5)
params = [(3,5), (13,26), (20, 31)]

for idx in range(len(timeframes) - 1):
    best = Best(-1_000_000.0, None)
  
    # Find the best parameter
    for param in params:
        strategy = rq.strategies.EMACrossover(*param)
        account = rq.run(feed, strategy, timeframe=timeframes[idx])
        equity = account.equity_value()
        if equity > best.equity:
            best = Best(equity, param)

    # Validate 
    strategy = rq.strategies.EMACrossover(*best.param)
    account = rq.run(feed, strategy, timeframe=timeframes[idx+1])
    equity = account.equity_value()
    print(f"param={best.param} training={best.equity:,.0f} testing={equity:,.0f}")
```


## Multi-run
A Multi-run samples a number of random timeframes and then runs a
back test on each of them.

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

```python
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

## Early stopping
It is possible to stop a run before all the events are handled. You do so
by having one of the components invoke the `stop_run()` function.

For example a custom Journal could track some metrics and based on the results
invoke the `stop_run()` function. See also [journal](journal.md#guard-journal)

