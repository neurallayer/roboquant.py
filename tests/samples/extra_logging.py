# %% [markdown]
# # Logging
# Sometimes it is useful yo increase roboquant logging level to get
# better insights into a run. Especially more complex logic like that in
# the `FlexTrader`, can be difficult to understand.

# %%
import logging
import roboquant as rq

logging.basicConfig()

# %% [markdown]
# Set logging at higher level
logging.getLogger("roboquant").setLevel(logging.WARNING)

# %% [markdown]
# Set logging level at individual module
rq.traders.flextrader.logger.setLevel(logging.INFO)

# %% [markdown]
# Run a badk test to see the results
feed = rq.feeds.YahooFeed("AAPL", "MSFT", start_date="2024-01-01", end_date="2024-08-01")
strategy = rq.strategies.EMACrossover()
trader = rq.traders.FlexTrader()
account = rq.run(feed, strategy, trader)
