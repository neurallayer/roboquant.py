# %% [hidden]
from matplotlib import dates, pyplot as plt
from matplotlib.dates import AutoDateLocator, AutoDateFormatter
import roboquant as rq
from roboquant.common.timeseries import TimeSeries
from roboquant.journals.metricsjournal import MetricsJournal

rq.set_dark_style()

# %% [markdown]
# First we need to get the data for our backtest. It should have sufficient
# historic data for each step in our walkforward.
# <br/><br/>
# In this example we get over 15 years of historic market data from Yahoo Finance

# %%
feed = rq.feeds.YahooFeed.us_stocks_10(start_date="2010-01-01")


# %% [markdown]
# We can now iterate over different hyper parameter values
# and capture the equity values for each run.
# %%
params = [(2,5), (5,7), (7,11), (9,17), (13,26), (20, 50), (30, 70)]
equities: dict[str, TimeSeries] = {}
for fast, slow in params:
    strategy = rq.strategies.EMACrossover(fast, slow)
    journal = MetricsJournal.pnl()
    account = rq.run(feed, strategy, journal=journal)
    equity = journal.get_metrics("pnl/equity")
    name = f"ema-{fast}/{slow}"
    equities[name] = equity

# %% [markdown]
# FInally, we plot each equity curve and can see the
# differences in result
# %%
z = 1.0
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
for label, equity in equities.items():
    # matplotlib doesn't support datetime on 3d charts
    x = [dates.date2num(d) for d in equity.index]
    y = equity["pnl/equity"]
    ax.plot(x, y, z, zdir="y", label=label)
    z += 1

ax.xaxis.set_major_formatter(AutoDateFormatter(AutoDateLocator()))
ax.xaxis.set_label_text("time")
ax.yaxis.set_label_text("run")
ax.zaxis.set_label_text("equity")
plt.legend();

# %%

# %%
