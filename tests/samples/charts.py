# %%
# uncomment the following line
# %pip install roboquant

# %% [markdown]
# This example shows how to draw certain charts using `roboquant`.
# It uses the `YahooFeed` to fetch historical data for several assets and then runs a
# simple EMA Crossover strategy. The results are visualized using the `matplotlib` library
# and the `roboquant` plotting capabilities.
# %%
import matplotlib.pyplot as plt
import roboquant as rq
import roboquant.journals.metrics
from roboquant.journals.report import Report
from roboquant.timeseries import TimeSeries

# Setup some defaults for matplotlib
plt.style.use("dark_background")
plt.rcParams['figure.figsize'] = [10.0, 5.0]
plt.rcParams['figure.dpi'] = 150

# %%
feed = rq.feeds.YahooFeed.us_stocks_10()

# %% [markdown]
# Plot a price chart for one of the assets in the feed
feed.plot("MSFT");

# %%
strategy = rq.strategies.EMACrossover()
journal = rq.journals.MetricsJournal.pnl()
account = rq.run(feed, strategy, journal=journal)


# % [markdown]
# ## Customize
# You can customize many of the plots by providing parameter arguments that will be passed on
# to matplotlib.
# %%
equity = journal.get_metric("pnl/equity")
ax = equity.plot(color="green", linewidth=1)
ax.set_title("My Custom Title");


# % [markdown]
# Or you can take full control of the ax and make more advanced chart figures.
# Below we create a figure with 10 subplots.
# We also create a more advanced title.
# %%
tf = rq.Timeframe.previous("365 days")
_, axs = plt.subplots(5, 2, figsize=(20, 30))

for ax, asset in zip(axs.flatten(), feed.assets()):
    asset_trades = account.trades_for_asset(asset)
    pnl = sum(trade.pnl for trade in asset_trades)
    ax = feed.plot(asset, timeframe=tf, ax=ax, trades=account.trades, linewidth=1)
    ax.grid(True, color="grey", linestyle="--")
    ax.set_title(f"{asset.symbol} ({pnl:,.0f} {asset.currency})")

# %% [markdown]
# ## Multi-run
# Perform a walk forward over 4 equal timeframes and
# - plot each run on the same chart.

# %%
timeframes = feed.timeframe().split(4)
ax = None

for timeframe in timeframes:
    strategy = rq.strategies.EMACrossover()
    journal = rq.journals.MetricsJournal.pnl()
    rq.run(feed, strategy, journal=journal, timeframe=timeframe)
    equity = journal.get_metric("pnl/equity")
    ax = equity.plot(ax=ax, linewidth=0.5)

# %% [markdown]
# Very similar to previous example, but now we
# plot a single equity curve.

# %%
timeframes = feed.timeframe().split(4)
ax = None
overlap = "35 days"

equities = []

for timeframe in timeframes:
    strategy = rq.strategies.EMACrossover()
    journal = rq.journals.MetricsJournal.pnl()
    rq.run(feed, strategy, journal=journal, timeframe=timeframe.prepend(overlap))
    equity = journal.get_metric("pnl/equity")
    equities.append(equity.pct_change())

single_ts = TimeSeries.concat(*equities).inverse_pct_change()
single_ts.plot();

# %% [markdown]
# Run randomly sampled 1-year back tests and plot the equity curve
# for each run on the same chart. This provides visual insights
# how the equity results are distributed.

# %%
timeframes = feed.timeframe().sample(200, "365 days")
ax = None

for timeframe in timeframes:
    strategy = rq.strategies.EMACrossover()
    journal = rq.journals.MetricsJournal.pnl()
    rq.run(feed, strategy, journal=journal, timeframe=timeframe)
    equity = journal.get_metric("pnl/equity")[26:]
    ax = equity.plot(plot_timeline=False, ax=ax, linewidth=0.5, color="grey", alpha=0.5)

# %% [markdown]
# Report enables to publication of mathplotlib charts. They can be saved
# as a single self-contained PDF file or HTML file.

# %%
strategy = rq.strategies.EMACrossover(26, 50)
journal = rq.journals.MetricsJournal.pnl()
account = rq.run(feed, strategy, journal=journal)

report = Report()
for asset in feed.assets():
    feed.plot(asset, trades=account.trades, linewidth=0.5, color="grey")
    report.add_current_figure()

journal.plot("pnl/equity")
report.add_current_figure()

df = account.trades_to_dataframe().round(2)
top_trades = df.sort_values("pnl", ascending=False)[:25]
report.add_df(top_trades, "top 25 winners")

down_trades = df.sort_values("pnl", ascending=True)[:25]
report.add_df(down_trades, "top 25 losers")

report.save_as_pdf("/tmp/report.pdf")
report.save_as_html("/tmp/report.html")

# %%
# Using the scorecard journal
strategy = rq.strategies.EMACrossover()
asset = feed.assets()[0]
scorecard = rq.journals.Scorecard(roboquant.journals.metrics.PNLMetric(), include_prices=True)
rq.run(feed, strategy, journal=scorecard)
scorecard.plot();


# %%
# Mix in some ETF's to have a more diverse set of assets.
feed = rq.feeds.YahooFeed("MSFT", "JPM", "XOM", "F", "GLD", "GSG", "BND", "LQD", "TIP", "IBIT", "VIXY")
feed.plot_corr(fontsize=7);


# %%
