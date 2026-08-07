---
kernelspec:
  name: python3
  display_name: Python 3
---

# Basic Charts

This page shows how to draw certain type of charts using *roboquant*.

It uses the `YahooFeed` to fetch historical data for several assets and then runs a
simple EMA Crossover strategy. The results are visualized using the `matplotlib` library
and the `roboquant` plotting capabilities.

:::{note}
Roboquant isn't designed to be a pure visual algo-trading tool. Charts are included to provide
insights into what is happening during a run but are not the basis for strategies.

There is no out-of-the-box support for candle stick charts and things like the drawing of support lines.
Although these can be easily added using third party packages like mplfinance, they are not the
focus area for roboquant.
:::

We start with importing the required packages and setting some defaults.

```{code-cell} python
import roboquant as rq
import matplotlib.pyplot as plt

# Setup some defaults for matplotlib
rq.set_dark_style()
```

Then we load the prices of 8 very different assets to make the results a bit
more interesting.

```{code-cell} python
feed = rq.feeds.YahooFeed("MSFT", "F", "GLD", "GSG", "BND", "LQD", "IBIT", "VIXY")
```

## Feed Chart

### Price Chart
Plot a price and optionally the volume for one of the assets in the feed.

```{code-cell} python
feed.plot("MSFT");
```

### Correlation Chart
Sometimes it is useful to inspect the correlation between the assets we want to trade in.
There is a special plot method available that makes this visibe.

```{code-cell} python
feed.to_timeseries().plot_corr(fontsize=7);
```

## Backtest Charts

Once we run a backtest we can plot the account related charts and charts 
for any metrics we captured.

In the code snippet below the `MetricsJournal.pnl()` will capture metrics like
the total **equity** at each step of the run.

```{code-cell} python
strategy = rq.strategies.EMACrossover()
journal = rq.journals.MetricsJournal.pnl()
account = rq.run(feed, strategy, journal=journal)
```

### Trade Chart

```{code-cell} python
tf = rq.Timeframe.previous("365 days")
feed.plot("MSFT", timeframe=tf, trades=account.trades);
```

### Equity Chart
Equity is a good example of a metric that is usefull to capture during a run.
It provides insights how the total equity is evolving during a run and shows 
also the volatility of our strategy when it comes to returns.

```{code-cell} python
equity = journal.get_metrics("pnl/equity")
equity.plot();
```

### Allocation Chart
For larger portfolios it is useful to see which percentage is allocated to which asset.

Since assets can be denoted in different currencies, roboquant takes care of converting them to a single
currency before plotting.

```{code-cell} python
_, ax = plt.subplots(figsize=(3, 3))
account.plot_allocation(include_cash=True, ax = ax);
```

### Custom layouts
You can customize many of the plots by providing parameter arguments that will be passed on
to matplotlib.

```{code-cell} python
equity = journal.get_metrics("pnl/equity")
ax = equity.plot(color="green")
ax.set_title("My Custom Title");
```

Or you can take full control of the layout and make more advanced chart figures.
Below we create a figure with 8 subplots. We also create a more informative title.

```{code-cell} python
tf = rq.Timeframe.previous("365 days")
_, axs = plt.subplots(4, 2, figsize=(20, 30))

for ax, asset in zip(axs.flatten(), feed.assets()):
    pnl = account.pnl(asset)
    ax = feed.plot(asset, timeframe=tf, ax=ax, trades=account.trades)
    ax.set_title(f"{asset.symbol} ({pnl:,.0f})")
```

## Multi-run
Rather that running a single back test, we can also run multiple back tests and plot the results on the same chart.
This is useful to see how the strategy performs over different timeframes or with different parameters.

One pattern we use to plot multiple runs on the same chart is:

```python
ax = None
for i in some_range:
    ...
    ax = something.plot(ax=ax, ....)
```

The first time the plot method is invoked, ax is None and the plot method will create a new figure and axis. 
It will return this axis, so that next time the plot method is invoked it will plot on the existing axis.

### Walk Forward

Perform a walk-forward over 4 equal timeframes and plot the equity curve of each run on the same chart.

```{code-cell} python
timeframes = feed.timeframe().split(4)
ax = None

for timeframe in timeframes:
    strategy = rq.strategies.EMACrossover()
    journal = rq.journals.MetricsJournal.pnl()
    rq.run(feed, strategy, journal=journal, timeframe=timeframe)
    equity = journal.get_metrics("pnl/equity")
    ax = equity.plot(ax=ax, legend=False)
```

### Sample Random Timeframes

Run randomly sampled 1-year back tests and plot the equity curve
for each run on the same chart. This provides visual insights
how the equity curves are distributed.

```{code-cell} python
timeframes = feed.timeframe().sample(100, "365 days")
ax = None

for timeframe in timeframes:
    strategy = rq.strategies.EMACrossover(5, 13)
    journal = rq.journals.MetricsJournal.pnl()
    rq.run(feed, strategy, journal=journal, timeframe=timeframe)

    # Skip the first 13 trading days since the strategy is still
    # warming up and the equity curve is flat during this period.
    equity = journal.get_metrics("pnl/equity")[13:]

    ax = equity.plot_without_timeline(ax=ax, linewidth=2, color="grey", alpha=0.2, legend=False)
```

