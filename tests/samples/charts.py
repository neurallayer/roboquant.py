# %% [markdown]
# This example shows how to draw certain charts using the `roboquant` library.
# It uses the `YahooFeed` to fetch historical data for several assets and then runs a
# simple EMA Crossover strategy. The results are visualized using the `matplotlib` library
# and the `roboquant` plotting capabilities.
# %%
import matplotlib.pyplot as plt
import matplotlib as mpl
import roboquant as rq
from datetime import timedelta
from roboquant.journals.report import Report

plt.style.use('dark_background')
mpl.rcParams['figure.facecolor'] = '#202020'

# %%
feed = rq.feeds.YahooFeed("JPM", "IBM", "F", start_date="2010-01-01")

# %% [markdown]
# Plot a price chart for each of the assets in the feed

# %%
for asset in feed.assets():
    feed.plot(asset, linewidth=0.5)

# %%
strategy = rq.strategies.EMACrossover()
journal = rq.journals.MetricsJournal.pnl()
account = rq.run(feed, strategy, journal=journal)
feed.plot("IBM", trades=account.trades, linewidth=0.5, color="grey")

# %%
equity = journal.get_metric("pnl/equity")
_ = equity.plot(color="green", linewidth=0.5)

# %%
# Perform a walk forward over 4 equal timeframes and plot each run on the same chart.

timeframes = feed.timeframe().split(4)
_, ax = plt.subplots()


for timeframe in timeframes:
    strategy = rq.strategies.EMACrossover()
    journal = rq.journals.MetricsJournal.pnl()
    rq.run(feed, strategy, journal=journal, timeframe=timeframe)
    equity = journal.get_metric("pnl/equity")
    equity.plot(ax=ax, linewidth=0.5)

# %%
# Run 50 1-year back tests and plot the equity curve for each run on the same chart.
# This provides insights how the results are distrubuted and what to expect.

one_year = timedelta(days=365)
timeframes = feed.timeframe().sample(one_year, 50)
_, ax = plt.subplots()

for timeframe in timeframes:
    strategy = rq.strategies.EMACrossover()
    journal = rq.journals.MetricsJournal.pnl()
    rq.run(feed, strategy, journal=journal, timeframe=timeframe)
    equity = journal.get_metric("pnl/equity")
    equity.plot(plot_x=False, ax=ax, linewidth=0.5, color="grey")


# %% [markdown]
# Report enables to publication of matlplotlib charts. They can be saved
# as a single self-contained PDF file or HTML file.

# %%
strategy = rq.strategies.EMACrossover()
journal = rq.journals.MetricsJournal.pnl()
account = rq.run(feed, strategy, journal=journal)

report = Report()
for asset in feed.assets():
    feed.plot(asset, trades = account.trades, linewidth=0.5, color="grey")
    report.add_current_figure()

for metric_name in journal.get_metric_names():
    journal.plot(metric_name)
    report.add_current_figure()

report.save_as_pdf("/tmp/report.pdf")
report.save_as_html("/tmp/report.html")

# %%
# Using the scorecard journal
strategy = rq.strategies.EMACrossover()
asset = feed.assets()[0]
scorecard = rq.journals.ScoreCard(rq.journals.PNLMetric(), include_prices=True)
rq.run(feed, strategy, journal=scorecard)
scorecard.plot(size=(8.27, 30), linewidth=0.5)


