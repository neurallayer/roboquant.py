# %% [markdown]
# This example shows how to use the Yahoo feed with a simple EMA Crossover strategy
# to run a backtest in roboquant.

# %%
import roboquant as rq
from roboquant.traders.fixedtrader import FixedTrader

# %%
feed = rq.feeds.YahooFeed.us_stocks_10()

# %%
strategy = rq.strategies.EMACrossover()
account = rq.run(feed, strategy)
print(account)

# %%
trader = FixedTrader(feed.assets())
account = rq.run(feed, None, trader = trader)
print(account)
