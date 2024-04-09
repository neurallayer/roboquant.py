# %%
import logging
import roboquant as rq

# %%
# Increase logging level to get better insights into the FlexTrader decision making 
logging.basicConfig()
logging.getLogger("roboquant.traders.flextrader").setLevel(logging.INFO)

symbols = rq.feeds.get_sp500_symbols()[10:30]
feed = rq.feeds.YahooFeed(*symbols, start_date="2000-01-01")
strategy = rq.strategies.EMACrossover()
account = rq.run(feed, strategy)
print(account)
