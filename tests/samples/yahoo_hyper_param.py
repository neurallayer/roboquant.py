# %% [hidden]
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
equities = TimeSeries()
for fast, slow in params:
    strategy = rq.strategies.EMACrossover(fast, slow)
    journal = MetricsJournal.pnl()
    account = rq.run(feed, strategy, journal=journal)
    equity = journal.get_metric("pnl/equity", f"ema-{fast}/{slow}")
    equities = equities.join(equity, how="outer")

# %%
ax = equities.plot_3d();

# %%
