# %%
import logging
import roboquant as rq

# %%
logging.basicConfig(level=logging.WARNING)
logging.getLogger("roboquant.traders.flextrader").setLevel(logging.INFO)

feed = rq.feeds.YahooFeed("JPM", "IBM", "TSLA", start_date="2000-01-01")
strategy = rq.strategies.EMACrossover()
account = rq.run(feed, strategy)
print(account)
