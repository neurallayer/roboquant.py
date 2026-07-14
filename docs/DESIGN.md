# Design Principles

## Modular Pipeline Architecture

The library is built around a clean separation of five orthogonal concerns:

| Component | Responsibility | Has access to Account? |
|-----------|---------------|----------------------|
| **Feed** | Provides market data events | No |
| **Strategy** | Generates trading signals from events | No |
| **Trader** | Converts signals into orders (risk/sizing) | Yes |
| **Broker** | Executes orders, maintains account state | Yes (owns Account) |
| **Journal** | Logs/tracks every step (read-only) | Read-only snapshot |

Each component is an abstract base class with pluggable implementations, making every part of the pipeline independently swappable.

## The Run Loop

The core of the system is the `roboquant.run()` function, which connects all components in a streaming event loop:

```
for each event in feed.play(timeframe):
    1. broker.sync(event)                   — update account, execute fills
    2. strategy.create_signals(event)       — generate signals from market data
    3. trader.create_orders(signals, ...)   — apply risk rules, produce orders
    4. broker.place_orders(orders)          — submit orders to the broker
    5. journal.track(event, account, ...)   — record metrics (optional)
```

### Step-by-step

1. **`broker.sync(event)`** — Updates the account with the latest market data. Open orders from previous steps are tested against prices and executed if conditions are met. No look-ahead bias: orders placed at time `t` only execute at time `t+1`.

2. **`strategy.create_signals(event)`** — The strategy examines the event's price data and returns a list of `Signal` objects. Each signal has an asset, a rating (typically -1.0 to 1.0), and a type (`ENTRY`, `EXIT`, or `ENTRY_EXIT`). Strategies are **pure decision-makers** — they know nothing about cash, positions, or risk.

3. **`trader.create_orders(signals, event, account)`** — The trader applies risk management rules (position sizing, shorting constraints, order limits) and converts signals into concrete `Order` objects. Unlike strategies, traders **have full access to the Account** (cash, positions, buying power).

4. **`broker.place_orders(orders)`** — New orders are submitted to the broker. In `SimBroker`, they are stored and evaluated for execution when the next event arrives.

5. **`journal.track(...)`** — Optional logging and metrics collection. Journals are passive observers that never modify state.

## Minimal Backtest

```python
import roboquant as rq

feed = rq.feeds.YahooFeed("JPM", "IBM", start_date="2015-01-01")
account = rq.run(feed, rq.strategies.EMACrossover())
print(account)
```

This works because `run()` provides sensible defaults: `SimBroker` (USD 1M deposit, 0% slippage) and `FlexTrader` (conservative position sizing).

## Custom Backtest

```python
feed = rq.feeds.YahooFeed("AAPL", "MSFT", start_date="2020-01-01")
strategy = rq.strategies.EMACrossover()
trader = rq.traders.FlexTrader(max_order_perc=0.1, shorting=True)
broker = rq.brokers.SimBroker(deposit=500_000)
journal = rq.journals.MetricsJournal()

account = rq.run(feed, strategy, trader=trader, broker=broker, journal=journal)
print(account)
print(journal.metrics())
```

## Strategy/Trader Separation

This is the most important design choice in the library:

- **Strategies** produce signals from market data only. They implement `create_signals(event) -> list[Signal]` and have no access to account state. This keeps them pure, testable, and reusable across any trader configuration.
- **Traders** implement `create_orders(signals, event, account) -> list[Order]` and are responsible for risk management, position sizing, and order construction. They know nothing about indicators or market data beyond what is in the signal.

Example: The same `EMACrossover` strategy can be used with a conservative trader (2% max order) in backtesting and a different trader (20% max order, shorting enabled) in live trading — without changing a line of strategy code.

## Key Principles

- **Event-driven streaming** — Everything is built around `Event` objects produced lazily by feeds, supporting both backtesting and live trading with the same pipeline.
- **Broker owns the Account** — The Account is never modified directly by user code; it is always the broker's canonical view returned by `sync()`.
- **Default-everywhere** — `roboquant.run(feed, strategy)` works out of the box with sensible defaults, making the simplest case a one-liner.
- **Immutable core types** — `Asset`, `Signal`, and `Event` are immutable; `Order` uses `cancel()`/`modify()` returning new objects.
- **Strategy composition** — `MultiStrategy` combines multiple strategies with configurable conflict resolution (`first`, `last`, `mean`, `none`).
- **Pluggable pricing** — Price type strings (`"OPEN"`, `"CLOSE"`, `"HIGH"`, `"LOW"`, `"DEFAULT"`) allow strategies and traders to choose which price to use for evaluation.
- **Multi-currency** — Built-in support via `Amount`, `Wallet`, and pluggable `CurrencyConverter` (ECB, static, one-to-one).
