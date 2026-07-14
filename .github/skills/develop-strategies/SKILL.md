---
name: develop-strategies
description: >-
  Skill that helps to write trading solutions using the roboquant platform in Python.
  Covers custom strategies, traders, backtesting, and the event pipeline.
---

# Develop Strategies

This skill helps you to create custom Python trading strategies using roboquant frmaework 
and then use these srategies in backtesting.

## Creating a Strategy

Subclass `Strategy` and implement `create_signals(event: Event) -> list[Signal]`:

```python
from roboquant import Signal, SignalType
from roboquant.strategies import Strategy

class MyStrategy(Strategy):
    def create_signals(self, event: Event) -> list[Signal]:
        result: list[Signal] = []
        for asset, price in event.get_prices("CLOSE").items():
            result.append(Signal.buy(asset))
        return result
```

### Signal API

| Constructor | rating | type | Use |
|---|---|---|---|
| `Signal.buy(asset)` | 1.0 | `ENTRY_EXIT` | Strong buy |
| `Signal.sell(asset)` | -1.0 | `ENTRY_EXIT` | Strong sell |
| `Signal(asset, rating, type)` | custom | custom | Full control |

`rating` is typically -1.0 to 1.0. `SignalType` can be `ENTRY`, `EXIT`, or `ENTRY_EXIT`.

### Accessing prices

```python
for asset, item in event.price_items.items():
    if isinstance(item, Bar):
        o, h, l, c, v = item.ohlcv  # open, high, low, close, volume

# Get all prices of a type:
prices: dict[Asset, float] = event.get_prices("CLOSE")

# Single asset:
price = event.get_price(asset, "OPEN")
```

See `PriceItem.price(price_type)` — `Bar` defaults to `CLOSE`, `Quote` defaults to `MID`, `TradePrice` uses its single price.

### Strategy state (per-asset tracking)

Track per-asset state in a dict:

```python
class MyStrategy(Strategy):
    def __init__(self, period=14):
        self.period = period
        self._prices: dict[Asset, list[float]] = {}

    def create_signals(self, event: Event) -> list[Signal]:
        result = []
        for asset, price in event.get_prices("CLOSE").items():
            prices = self._prices.setdefault(asset, [])
            prices.append(price)
            if len(prices) >= self.period:
                ...
        return result
```

For large lookbacks, use `OHLCVBuffers` from `roboquant.strategies.buffer` (pre-allocated numpy arrays):

```python
from roboquant.strategies.buffer import OHLCVBuffers

class MyStrategy(Strategy):
    def __init__(self):
        self.buffers = OHLCVBuffers(50)

    def create_signals(self, event: Event) -> list[Signal]:
        ready = self.buffers.add_event(event)
        for asset in ready:
            closes = self.buffers[asset].close()  # NDArray[np.float64]
            ...
```

### Entry vs exit signals

```python
Signal.buy(asset, SignalType.ENTRY)       # only opens/increases
Signal.sell(asset, SignalType.EXIT)       # only closes/reduces
Signal.buy(asset, SignalType.ENTRY_EXIT)  # both (default)
```

`FlexTrader` respects `signal.is_entry` / `signal.is_exit`.

### Strategy composition

Combine strategies with `MultiStrategy`:

```python
combined = MultiStrategy(s1, s2, signal_filter="mean")
# Filters: "first", "last", "mean", "none" (default)
```

Wrap with `CachedStrategy` for repeated runs:

```python
cached = CachedStrategy(my_strategy, feed)
```

## Creating a Trader

Traders convert signals to orders and have access to the `Account`:

```python
from roboquant.traders import Trader
from roboquant import Order

class MyTrader(Trader):
    def create_orders(self, signals, event, account) -> list[Order]:
        orders = []
        for s in signals:
            price = event.get_price(s.asset)
            if price is None:
                continue
            orders.append(Order(s.asset, Decimal(100), price, "DAY"))
        return orders
```

### FlexTrader (default, configurable)

```python
trader = FlexTrader(
    one_order_only=True,       # no duplicate orders per asset
    size_fractions=0,          # fractional share digits
    max_order_perc=0.05,       # 5% of equity per order
    min_order_perc=0.02,       # 2% of equity minimum
    max_position_perc=0.1,     # 10% max position size
    safety_margin_perc=0.05,   # reserved buying power
    shorting=False,            # disallow short positions
    limit_offset_perc=0.0,     # offset limit price from market
    tif="DAY",                 # "DAY" or "GTC"
)
```

Debug dropped signals:

```python
logging.getLogger("roboquant.traders.flextrader").setLevel(logging.INFO)
```

## Order API

```python
Order(asset, size, limit, tif, **kwargs)
# size:  Decimal (positive=buy, negative=sell)
# limit: float
# tif:   "DAY" | "GTC"

# Cancel or modify:
cancel = order.cancel()
mod    = order.modify(size=Decimal(50), limit=150.0)
```

## Running a Backtest

```python
import roboquant as rq

feed = rq.feeds.YahooFeed("AAPL", "MSFT", start_date="2020-01-01")
account = rq.run(feed, rq.strategies.EMACrossover())
```

### Customised

```python
strategy = rq.strategies.IBSStrategy(0.3, 0.7)
trader = rq.traders.FlexTrader(max_order_perc=0.1, shorting=True)
broker = rq.brokers.SimBroker(deposit=500_000, price_type="CLOSE", slippage=0.001)
journal = rq.journals.MetricsJournal(rq.journals.PNLMetric())

account = rq.run(feed, strategy, trader=trader, broker=broker, journal=journal)
```

### Walk-forward

```python
for tf in feed.timeframe().split(4):
    account = rq.run(feed, strategy, timeframe=tf)
    print(tf, account.equity())
```

### SimBroker

```python
broker = rq.brokers.SimBroker(
    deposit=1_000_000 @ rq.monetary.USD,
    price_type="OPEN",
    slippage=0.001,
)
broker.reset()  # reuse for another run
```

### Journals

```python
rq.journals.BasicJournal()                              # counts only
rq.journals.MetricsJournal(rq.journals.PNLMetric())     # time series
rq.journals.ScoreCard()                                 # final summary
```

### Key concepts

- Pipeline: `Feed → Event → Strategy → Signal → Trader → Order → Broker → Account`
- Orders placed at time `t` execute at `t+1` (no look-ahead bias)
- `Strategy` has no account access; `Trader` does
- `Broker.sync(event)` processes fills and returns updated account
- The returned `Account` holds final positions, trades, cash, buying power, equity
- Signal ratings propagate to order sizing in `FlexTrader`
