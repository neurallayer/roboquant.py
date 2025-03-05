# %%
import matplotlib.pyplot as plt
import matplotlib as mpl
import roboquant as rq
from datetime import timedelta

plt.style.use('dark_background')
mpl.rcParams['figure.facecolor'] = '#202020'
# %%
feed = rq.feeds.YahooFeed("JPM", "IBM", "F", start_date="2010-01-01")

for asset in feed.assets():
    feed.plot(asset)

# %%
strategy = rq.strategies.EMACrossover()
journal = rq.journals.MetricsJournal.pnl()
rq.run(feed, strategy, journal=journal)

# %%
journal.plot("pnl/equity", color="green", linewidth=0.5)

# %%
# Perform a walk forward over 4 equal timeframes and plot each run.

timeframes = feed.timeframe().split(4)
_, ax = plt.subplots()

for timeframe in timeframes:
    strategy = rq.strategies.EMACrossover()
    journal = rq.journals.MetricsJournal.pnl()
    rq.run(feed, strategy, journal=journal, timeframe=timeframe)
    journal.plot("pnl/equity", ax=ax, linewidth=0.5)

# %%
# Run 50 1-year back tests and plot the equity curve for each run.
# This provides insights how the results are distrubuted and what to expect.

one_year = timedelta(days=365)
timeframes = feed.timeframe().sample(one_year, 50)
_, ax = plt.subplots()

for timeframe in timeframes:
    strategy = rq.strategies.EMACrossover()
    journal = rq.journals.MetricsJournal.pnl()
    rq.run(feed, strategy, journal=journal, timeframe=timeframe)
    journal.plot("pnl/equity", plot_x=False, ax=ax, linewidth=0.5, color="grey")
