# %%
import matplotlib.pyplot as plt
import matplotlib as mpl
import roboquant as rq

plt.style.use('dark_background')
mpl.rcParams['figure.facecolor'] = '#202020'
# %%
feed = rq.feeds.YahooFeed("JPM", "IBM", "F", start_date="2010-01-01")
_, axs = plt.subplots(2, figsize=(10, 10))

feed.plot(axs[0], feed.assets()[0])
feed.plot(axs[1], feed.assets()[1])
plt.show()

# %%
strategy = rq.strategies.EMACrossover()
journal = rq.journals.MetricsJournal.pnl()
rq.run(feed, strategy, journal=journal)

# %%
journal.plot(plt, "pnl/equity", color="green", linewidth=0.5)
plt.show()

# %%
metric_names = journal.get_metric_names()
_, axs = plt.subplots(3, 3, figsize=(15, 15))
axs = axs.flatten()
for ax, metric_name in zip(axs, metric_names):
    journal.plot(ax, metric_name)
plt.show()

# %%
timeframes = feed.timeframe().split(4)

for timeframe in timeframes:
    strategy = rq.strategies.EMACrossover()
    journal = rq.journals.MetricsJournal.pnl()
    rq.run(feed, strategy, journal=journal, timeframe=timeframe)
    journal.plot(plt, "pnl/equity", linewidth=0.5)
