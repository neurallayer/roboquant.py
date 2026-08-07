---
kernelspec:
  name: python3
  display_name: Python 3
---

# Time & TimeSeries
Time in roboquant uses the Python `datetime` object with timezone set to UTC.

For example `event.time` is always in timezone UTC, even if the event orginates
from an exchange in another timezone. 

## Timeframe
A timeframe represents a period in time with a certain start- and end-time. Like most
time variables, these are Python `datetime` objects using the UTC timezone.

The start-time is always inclusive, but the end-time can be either inclusive or exclusive.

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
TimeSeries represents a multi-variate timeseries. It extends Pandas DataFrame
with the index always being a timeline and the columns are always float values.

