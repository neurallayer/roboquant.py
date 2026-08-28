---
kernelspec:
  name: python3
  display_name: Python 3
---

# Time & TimeSeries

## Overview
Time related data in *roboquant* uses the Python `datetime` object with
teh timezone set to UTC.

For example `event.time` is always in timezone UTC, even if the event originates
from an exchange in a different timezone. 

## Timeframe
(timeframe_def)=
A timeframe represents a period in time with a certain start- and end-time. Like other
time variables in *roboquant*, these are Python `datetime` objects using the UTC timezone.

The start-time of a timeframe is always inclusive, but the end-time can be either
inclusive or exclusive.

```{code-cell} python
import roboquant as rq

tf = rq.Timeframe.fromisoformat("2020-01-01", "2024-01-01", inclusive = True)
print(tf)

tf = rq.Timeframe.fromisoformat(
  "2021-01-01T00:12:00+00:00", 
  "2021-10-01T00:13:00+00:00",
  False)
print(tf)
```

You can split timeframes as well as sample from a timeframe, useful in certain 
types of back test.

```{code-cell} python
tfs = tf.split(5)
assert len(tfs) == 5

tfs = tf.sample(100, "60 days")
assert len(tfs) == 100
```

## Timeline
Timeline is not its own type but just defined as `list[datatime]`.


## TimeSeries
(timeseries_def)=
TimeSeries implements a multi-variate timeseries. It extends Pandas DataFrame
with the index always being a timeline and the columns are always float values.

```{code-cell} python
:tags: [hide-output]
import pandas as pd
import roboquant as rq

feed = rq.feeds.YahooFeed("IBM", start_date="2020-01-01")
df = feed.to_timeseries(rq.Stock("IBM"))
print("IBM Stock prices", df, sep="\n")

feed = rq.feeds.YahooFeed("IBM", "JPM", "MSFT", "TSLA", "INTC", start_date="2020-01-01")
data = feed.to_timeseries(*feed.assets())
print("Asset correlations:\n", data.corr())

strategy = rq.strategies.EMACrossover()
journal = rq.journals.MetricsJournal.pnl()
account = rq.run(feed, strategy, journal=journal)
equity = journal.get_metrics("pnl/equity")
print("Equity", equity, sep="\n")
```

:::{note}
There is also support for plain Pandas DataFrames in a few places. These
don't return timeseries but a dataframe representing a list of Python
objects like orders, trades and positions.

```{code-cell} python
:tags: [hide-output]
print(account.positions_to_dataframe())
print(account.orders_to_dataframe())
print(account.trades_to_dataframe())
```

:::