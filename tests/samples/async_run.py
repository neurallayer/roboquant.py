#%%
import roboquant as rq
from roboquant.alpaca import AlpacaLiveFeed
import os
from dotenv import load_dotenv
load_dotenv()

#%%
feed = rq.feeds.YahooFeed("AAPL", "IBM")
strategy = rq.strategies.EMACrossover()
account = rq.run(feed, strategy)
print(account)

#%%


api_key = os.environ["ALPACA_API_KEY"]
secret_key = os.environ["ALPACA_SECRET"]
feed = AlpacaLiveFeed(api_key, secret_key, market="iex")

feed.subscribe_bars("AAPL", "IBM")
strategy = rq.strategies.EMACrossover()
account = rq.run(feed, strategy)
print(account)
