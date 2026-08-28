---
name: roboquant-python-algo-trading
description: Develop, test, and improve algorithmic-trading strategies in the Python roboquant library with an AI coding agent. Use when creating roboquant strategies, custom traders, feeds, brokers, journals, indicators, backtests, or research workflows. This skill is specifically for the Python roboquant project, not the Kotlin version.
compatibility: Python 3.12+ recommended for the current roboquant.py main branch; requires the roboquant Python package and access to market-data feeds used by the project.
metadata:
  library: roboquant
  language: python
  focus: algorithmic-trading
  source: https://github.com/neurallayer/roboquant.py
---

# Roboquant Python algorithmic-trading development

Use this skill for the **Python** implementation of roboquant. Do not import Kotlin APIs, Kotlin examples, or Kotlin class names into Python code. The current Python repository is `neurallayer/roboquant.py`; the user-provided `roboquant/roboquant.py` location corresponds to that project lineage.

## Source-of-truth rule

Before inventing an API, inspect the installed roboquant package and, when available, the current repository source. Prefer the exact signatures in the current Python source over memories, third-party examples, or documentation for the Kotlin project.

Useful references:

- Python docs: https://python.roboquant.org/
- Python repository: https://github.com/neurallayer/roboquant.py
- PyPI: https://pypi.org/project/roboquant/

The current repository describes roboquant as an open-source Python algorithmic-trading platform. Its public design is event-driven: feeds produce events, strategies create signals, traders turn signals into orders, brokers execute/simulate them, and journals can record each step.

## Installation and project setup

Use an isolated environment and pin the roboquant version for reproducible research.

Typical setup:

```bash
python -m venv .venv
# activate .venv
python -m pip install --upgrade pip
python -m pip install roboquant
```

For the current repository, Python 3.12+ is the safest target. If the project already has a `pyproject.toml`, `uv.lock`, or other environment definition, use the project's existing environment rather than replacing it.

Before coding, verify:

```bash
python -c "import roboquant as rq; print(rq.__file__)"
python -c "import importlib.metadata as m; print(m.version('roboquant'))"
```

For a repository checkout, also run the project's existing verification/tests before changing behavior. Do not silently upgrade roboquant during a strategy experiment.

## The roboquant execution model

The normal backtest pipeline is:

1. A `Feed` yields `Event` objects.
2. A `Strategy` receives each event and creates zero or more `Signal` objects.
3. A `Trader` receives signals, the event, and the latest `Account`, and creates zero or more `Order` objects.
4. A `Broker` places those orders and synchronizes the account.
5. An optional `Journal` records the event, account, signals, and orders.

The current `roboquant.run()` implementation follows this sequence and returns the final `Account`.

A minimal backtest is:

```python
import roboquant as rq

feed = rq.feeds.YahooFeed("JPM", "IBM", "F", "TSLA")
strategy = rq.strategies.EMACrossover()

account = rq.run(feed, strategy)
print(account)
```

For custom strategies, keep the strategy deterministic and testable. Avoid hidden global state and avoid accessing future data.

## Which ABCs should be implemented?

The current Python source defines these important extension ABCs:

### 1. `Strategy` — implement this first for most trading ideas

Source: `roboquant/strategies/strategy.py`

Required method:

```python
from abc import ABC, abstractmethod
from roboquant.common.event import Event
from roboquant.common.signal import Signal

class Strategy(ABC):
    @abstractmethod
    def create_signals(self, event: Event) -> list[Signal]:
        ...
```

A `Strategy` is responsible for **signal generation**, not order construction.

Use it when the trading idea can be expressed as:
- inspect the current event/market data;
- calculate indicators or model features;
- emit zero or more `Signal` objects.

Do not put broker-specific execution logic into `Strategy`.

Typical shape:

```python
import roboquant as rq

class MyStrategy(rq.strategies.Strategy):
    def create_signals(self, event: rq.Event) -> list[rq.Signal]:
        signals: list[rq.Signal] = []
        # Inspect event.items and calculate the strategy decision.
        # Append rq.Signal objects when there is an actionable decision.
        return signals
```

Inspect the current `Signal` constructor and its enum/type values before writing signal creation code; do not guess field names.

### 2. `Trader` — implement when you need custom order/risk logic

Source: `roboquant/traders/trader.py`

Required method:

```python
from abc import ABC, abstractmethod
from roboquant.common.account import Account
from roboquant.common.event import Event
from roboquant.common.order import Order
from roboquant.common.signal import Signal

class Trader(ABC):
    @abstractmethod
    def create_orders(
        self,
        signals: list[Signal],
        event: Event,
        account: Account,
    ) -> list[Order]:
        ...
```

A `Trader` has access to the latest `Account`, unlike a `Strategy`. This makes it the appropriate place for:
- position-aware sizing;
- position exposure limits;
- converting signals into orders;
- reacting to current positions or cash;
- custom order-management rules.

Prefer a custom `Trader` over putting position state and order construction into `Strategy`.

If the standard trader behavior is sufficient, do not implement this ABC.

### 3. `Broker` — implement only for a new execution backend

Source: `roboquant/brokers/broker.py`

Required methods:

```python
from abc import ABC, abstractmethod
from roboquant.common.account import Account
from roboquant.common.event import Event
from roboquant.common.order import Order

class Broker(ABC):
    @abstractmethod
    def place_orders(self, orders: list[Order]) -> None:
        ...

    @abstractmethod
    def sync(self, event: Event | None = None) -> Account:
        ...
```

Implement `Broker` when integrating an execution venue or creating a specialized simulator.

Do not implement a broker merely to backtest a strategy. The normal backtest path uses `SimBroker()` when no broker is supplied.

A broker must preserve the roboquant contract:
- `place_orders()` accepts zero or more orders;
- a new order receives an ID from the broker;
- an existing order with zero size represents cancellation;
- an existing order with non-zero size represents an update;
- `sync()` returns the latest `Account`.

Test order placement, cancellation/update behavior, fills, cash, positions, fees, and rejected orders separately from strategy tests.

### 4. `Feed` — implement for a new market-data source

Source: `roboquant/feeds/feed.py`

Required methods:

```python
from abc import ABC, abstractmethod
from collections.abc import Iterator
from roboquant.common.asset import Asset
from roboquant.common.event import Event
from roboquant.common.timeframe import Timeframe

class Feed(ABC):
    @abstractmethod
    def play(self, timeframe: Timeframe | None = None) -> Iterator[Event]:
        ...

    @abstractmethod
    def assets(self) -> list[Asset]:
        ...
```

Implement a custom `Feed` only when the required data source is not already supported.

The feed must:
- yield events in chronological order;
- honor the requested timeframe when practical;
- expose the assets represented by the feed;
- use the same event model for backtesting and, where appropriate, live use;
- avoid leaking future information into earlier events.

### 5. `Journal` — implement for custom run logging/analysis

Source: `roboquant/journals/journal.py`

Required method:

```python
from abc import ABC, abstractmethod
from roboquant.common.account import Account
from roboquant.common.event import Event
from roboquant.common.order import Order
from roboquant.common.signal import Signal

class Journal(ABC):
    @abstractmethod
    def track(
        self,
        event: Event,
        account: Account,
        signals: list[Signal],
        orders: list[Order],
    ) -> None:
        ...
```

Implement a custom `Journal` when you need persistent audit data, custom metrics, experiment logging, or a specialized research output.

Do not use a custom Journal to change trading behavior. It should observe the run.

## Choosing the smallest correct extension

Use this decision rule:

- New trading idea -> subclass `Strategy`.
- New signal-to-order/risk/sizing policy -> subclass `Trader`.
- New exchange, execution API, or execution simulator -> subclass `Broker`.
- New market-data provider or event source -> subclass `Feed`.
- New logging/experiment/audit output -> subclass `Journal`.
- If an existing implementation already satisfies the requirement, configure/reuse it instead of implementing an ABC.

For a normal strategy project, the expected custom surface is usually **one `Strategy` subclass**, plus tests and a backtest runner.

## Backtesting workflow

Every strategy implementation should have a repeatable backtest entry point.

### Step 1: define the hypothesis

Write down before coding:
- assets/universe;
- data frequency;
- historical period;
- entry rules;
- exit rules;
- position-sizing rule;
- transaction costs/slippage assumptions;
- maximum exposure;
- benchmark;
- metrics used for acceptance/rejection.

Do not optimize a strategy against a single headline metric.

### Step 2: make a small smoke backtest

Start with a short date range and one or a few liquid assets.

Example:

```python
import roboquant as rq

feed = rq.feeds.YahooFeed(
    "AAPL",
    start_date="2023-01-01",
    end_date="2024-01-01",
)

strategy = MyStrategy()

account = rq.run(feed, strategy)
print(account)
```

Use the exact feed constructor available in the installed roboquant version; feed APIs can change.

### Step 3: run a full historical backtest

Use the longest clean history appropriate for the hypothesis. Keep the data source and version recorded.

For a simple strategy:

```python
account = rq.run(feed, strategy)
```

For custom components:

```python
account = rq.run(
    feed,
    strategy,
    trader=my_trader,
    broker=my_broker,
    journal=my_journal,
    timeframe=my_timeframe,
)
```

`run()` returns the latest `Account`. A journal is useful when the final account alone is insufficient to explain why trades occurred.

The current `run.py` source defaults the broker to `SimBroker()` when no broker is supplied. The current implementation also derives a `SimpleTrader` when no trader is supplied; do not rely on an older docstring that describes the default as `FlexTrader`.

### Step 4: validate the simulation

A backtest is not credible just because it runs.

Check:
- no look-ahead bias;
- indicators use only data available at the decision time;
- corporate actions/splits/dividends are handled appropriately by the selected data source;
- timestamps and time zones are correct;
- missing bars and duplicate events are understood;
- orders are generated only from available information;
- position sizes obey cash and exposure limits;
- commissions, spread, slippage, and other costs are represented where relevant;
- shorting/leverage behavior matches the intended strategy;
- the benchmark is calculated on the same period.

### Step 5: use out-of-sample validation

Prefer a time-ordered research process:

1. Train/choose parameters on an in-sample period.
2. Freeze the strategy and parameters.
3. Run on a later validation period.
4. Reserve a final untouched test period.
5. Compare with a sensible benchmark and simple baselines.

For parameter searches, avoid selecting the configuration solely because it has the highest in-sample return.

### Step 6: inspect more than return

At minimum inspect:
- total/annualized return;
- maximum drawdown;
- volatility;
- Sharpe or another risk-adjusted measure;
- number of trades;
- win/loss characteristics;
- turnover;
- exposure;
- transaction costs;
- performance by asset and time period.

If a metric or chart helper exists in the installed roboquant version, use it rather than reimplementing the calculation unnecessarily.

## Testing strategy code

Create fast unit tests that do not require network data.

Prefer deterministic synthetic or fixture events and test:
- no signal when conditions are false;
- exact signal when conditions become true;
- exits and reversals;
- indicator warm-up behavior;
- multiple assets;
- missing/invalid inputs;
- repeated events;
- boundary conditions;
- position sizing separately from signal generation.

Then add an integration test that runs a small deterministic feed through `rq.run()` with a simulated broker.

The AI coding agent should run the repository's existing test and lint commands rather than inventing a new test command. For a checkout using the current project's tooling, inspect `pyproject.toml`, `BUILD.md`, and CI configuration first.

## AI coding-agent workflow

When asked to create or modify a roboquant strategy:

1. Inspect the existing project structure and dependency/version configuration.
2. Inspect the installed roboquant API or repository source for the exact classes and signatures.
3. Identify whether the task needs `Strategy`, `Trader`, `Broker`, `Feed`, or `Journal`.
4. Implement the smallest extension that satisfies the requirement.
5. Add deterministic unit tests for the new behavior.
6. Add or update a reproducible backtest entry point.
7. Run lint/type checks and tests.
8. Run a smoke backtest.
9. Run the intended historical backtest.
10. Report assumptions, data range, costs, benchmark, metrics, and any known limitations.
11. Never claim that a profitable backtest proves future profitability.

When modifying an existing strategy, preserve its public API unless the task explicitly asks for a breaking change.

## Avoid these common agent mistakes

### Do not mix Python and Kotlin roboquant

Bad:
```text
Use Kotlin's Strategy interface, Roboquant class, Policy, or Kotlin coroutines in a Python strategy.
```

Good:
```text
Use roboquant.strategies.Strategy and implement create_signals().
```

The Python repository is a separate implementation even though the projects share concepts.

### Do not make Strategy create Orders

`Strategy` creates `Signal` objects. `Trader` creates `Order` objects. Keep that separation unless deliberately implementing the no-strategy path where all logic lives in a `Trader`.

### Do not implement every ABC

Most users need only `Strategy`. Customizing the framework unnecessarily increases complexity and makes backtests harder to trust.

### Do not use future data

Never calculate a signal using a bar, quote, trade, fundamental value, or derived feature that was not available at the event's decision timestamp.

### Do not optimize and evaluate on the same data

Separate parameter selection from final evaluation. Prefer walk-forward or rolling validation for strategies whose parameters are expected to adapt.

### Do not treat backtest output as financial advice

Backtested results are historical simulations, not guarantees. Explicitly report assumptions and limitations, especially around costs, liquidity, slippage, corporate actions, and data quality.

## Minimal project pattern

A practical project can look like:

```text
my-roboquant-strategy/
├── pyproject.toml
├── src/
│   └── my_strategy/
│       ├── __init__.py
│       └── strategy.py
├── tests/
│   └── test_strategy.py
└── backtests/
    └── run_backtest.py
```

Keep strategy logic separate from:
- data acquisition;
- experiment configuration;
- plotting/reporting;
- live deployment;
- credentials.

This makes it possible to reuse the same strategy logic with different feeds and brokers.

## Final acceptance checklist

Before declaring a roboquant strategy complete:

- [ ] Confirm this is the Python roboquant package/repository, not Kotlin roboquant.
- [ ] Confirm the installed/version-controlled roboquant API.
- [ ] Implement only the necessary ABC(s).
- [ ] Keep `Strategy -> Signal -> Trader -> Order -> Broker` responsibilities clear.
- [ ] Add deterministic unit tests.
- [ ] Run a smoke backtest.
- [ ] Run a full backtest over a documented period.
- [ ] Check for look-ahead/data leakage.
- [ ] Include realistic execution costs where relevant.
- [ ] Compare with a benchmark/baseline.
- [ ] Use out-of-sample or walk-forward validation for parameterized strategies.
- [ ] Record parameters, data period, feed, broker configuration, and results.
- [ ] Report drawdown and risk metrics, not just return.
- [ ] Do not claim that historical performance guarantees future performance.
