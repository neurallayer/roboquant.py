# %%
# Increase roboquant logging level to get better insights into a run
import logging
import roboquant as rq

logging.basicConfig()

# Set logging at higher level
logging.getLogger("roboquant").setLevel(logging.INFO)

# Set logging level at individual module
rq.traders.flextrader.logger.setLevel(logging.WARNING)

# %%
feed = rq.feeds.YahooFeed("AAPL", "MSFT", start_date="2022-01-01")
strategy = rq.strategies.EMACrossover()
account = rq.run(feed, strategy)
print(account)
