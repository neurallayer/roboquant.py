# %% [markdown]
# This example shows the different traders that come with roboquant.

# %%
import roboquant as rq
from roboquant.traders.buyholdtrader import BuyHoldTrader
from roboquant.traders.flextrader import FlexTrader
from roboquant.traders.simpletrader import SimpleTrader

# %%
feed = rq.feeds.YahooFeed.us_stocks_10()

# %%
trader = FlexTrader()
strategy = rq.strategies.EMACrossover()
account = rq.run(feed, strategy, trader=trader)
print(account)

# %%
trader = BuyHoldTrader(feed.assets())
strategy = None
account = rq.run(feed, strategy, trader = trader)
print(account)

# %%
trader = SimpleTrader()
strategy = rq.strategies.EMACrossover()
account = rq.run(feed, strategy, trader = trader)
print(account)
