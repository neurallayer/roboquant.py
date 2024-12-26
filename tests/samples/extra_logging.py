# %%
# Increase roboquant logging level to get better insights into a run
import logging
import roboquant as rq

logging.basicConfig()
logging.getLogger("roboquant").setLevel(logging.INFO)

# %%
feed = rq.feeds.YahooFeed("AAPL", "MSFT", start_date="2022-01-01")
strategy = rq.strategies.EMACrossover()
account = rq.run(feed, strategy)
print(account)
