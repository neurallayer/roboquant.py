# %%
import roboquant as rq

# %%
feed = rq.feeds.YahooFeed("JPM", "IBM", "F", start_date="2000-01-01")
strategy = rq.strategies.EMACrossover()
account = rq.run(feed, strategy)
print(account)
