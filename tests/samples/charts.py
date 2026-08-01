# %%
# uncomment the following line if you didn't install roboquant yet on your environment.
# %pip install --quiet roboquant

# %% [markdown]
# This example shows how to draw certain charts using `roboquant`.
# It uses the `YahooFeed` to fetch historical data for several assets and then runs a
# simple EMA Crossover strategy. The results are visualized using the `matplotlib` library
# and the `roboquant` plotting capabilities.
# %%
import roboquant as rq
import matplotlib.pyplot as plt

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
print(account)

# %%
account.plot_allocation(include_cash=True);


# %% [markdown]
# ## Customize
# You can customize many of the plots by providing parameter arguments that will be passed on
# to matplotlib.
# %%
equity = journal.get_metrics("pnl/equity")
ax = equity.plot(color="green", linewidth=1)
ax.set_title("My Custom Title");

# %% [markdown]
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
    equity = journal.get_metrics("pnl/equity")
    label = f"{timeframe.strftime('%Y-%m')}"
    ax = equity.plot(ax=ax, linewidth=0.5, label=label)

assert ax is not None
ax.legend(prop={'size': 5})

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
    equity = journal.get_metrics("pnl/equity")[26:]
    ax = equity.plot(plot_timeline=False, ax=ax, linewidth=0.5, color="grey", alpha=0.5)


# %% [markdown]
# ## Correlation
# Sometimes it is useful to inspect the correlation between assets.
# There is a special plot that makes this visibe.

# %%
# Mix in some ETF's to have a more diverse set of assets.
feed = rq.feeds.YahooFeed("MSFT", "JPM", "XOM", "F", "GLD", "GSG", "BND", "LQD", "TIP", "IBIT", "VIXY")
feed.to_timeseries().plot_corr(fontsize=7);
