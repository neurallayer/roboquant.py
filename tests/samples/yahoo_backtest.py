# %%
import roboquant as rq

# %%
feed = rq.feeds.YahooFeed("JPM", "IBM", "TSLA", "F", "INTC", start_date="2015-01-01")
strategy = rq.strategies.EMACrossover()
account = rq.run(feed, strategy)
print(account)
