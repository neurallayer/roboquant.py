---
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# Multi-process
One of the downsides of Python is that by default most of the computation is single-threaded.
However by using multiple processes, it is still possible to utilize all the cores available on your
machine (typically at the cost of higher memory usage).

This example shows how to perform a walk-forward using the `multiprocessing` package that comes with Python.
Each run is over acertain timeframe and set of parameters for the EMA Crossover strategy.

```{code-cell} python
from multiprocessing import get_context
from itertools import product

import roboquant as rq
```

```{code-cell} python
# Feed with over 25 years of data
feed = rq.feeds.YahooFeed.us_stocks_10(start_date="2000-01-01")
```

```{code-cell} python
def walk_forward(params: tuple[rq.Timeframe, tuple[int, int]]) -> str:
    """Perform a run over the provided timeframe and EMA parameters
    The return value is the equity value at the end of the run. In general,
    the return value needs to be serialized to be able to pass it back to the
    main process.
    """
    timeframe, (fast, slow) = params
    strategy = rq.strategies.EMACrossover(fast, slow)
    acc = rq.run(feed, strategy, timeframe=timeframe)
    result = f"{timeframe} EMA({fast:2},{slow:2}) ==> {acc.equity():.0f}"
    return result
```

Using "fork" ensures that the `feed` object is not being recreated for each process.
The pool is created with default number of processes (equal to the number of CPU cores).

```{code-cell} python
with get_context("fork").Pool() as p:

    # Split overall timeframe into 5 equal non-overlapping timeframes
    timeframe_params = feed.timeframe().split(5)

    # EMACrossover parameters, the fast and slow periods
    ema_params = [(3, 5), (5, 7), (10, 15), (15, 21)]

    # All the combinations of parameters (Cartesian product)
    all_params = list(product(timeframe_params, ema_params))

    assert len(all_params) == len(timeframe_params) * len(ema_params)

    # run the walk-forwards in parallel
    results = p.map(walk_forward, all_params)

    for row in results:
        print(row)
```
