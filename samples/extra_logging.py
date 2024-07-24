# %%
import logging
import roboquant as rq

# %%
# Increase logging level to get better insights into the strategy decision making 
logging.basicConfig()
logging.getLogger("roboquant.strategies").setLevel(logging.INFO)

feed = rq.feeds.YahooFeed("AAPL", "MSFT", start_date="2000-01-01")
strategy = rq.strategies.EMACrossover()
account = rq.run(feed, strategy)
print(account)
