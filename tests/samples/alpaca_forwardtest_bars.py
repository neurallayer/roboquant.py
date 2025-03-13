# %%
import os
import logging
import roboquant as rq
from roboquant.alpaca import AlpacaLiveFeed
from dotenv import load_dotenv
load_dotenv()

# %%
logging.basicConfig()
logging.getLogger("roboquant").setLevel(level=logging.INFO)

# Connect to Alpaca and subscribe to some IEX stocks 1-minute bars
symbols = ["TSLA", "MSFT", "NVDA", "AMD", "AAPL", "AMZN"]
api_key = os.environ["ALPACA_API_KEY"]
secret_key = os.environ["ALPACA_SECRET"]
alpaca_feed = AlpacaLiveFeed(api_key, secret_key, market="iex")
alpaca_feed.subscribe_bars(*symbols)

feed = rq.feeds.TimeGroupingFeed(alpaca_feed, 10.0)
# %%
# Let run an EMACrossover strategy
strategy = rq.strategies.EMACrossover(5, 13)
timeframe = rq.Timeframe.next(minutes=60)
journal = rq.journals.BasicJournal()
account = rq.run(feed, strategy, journal=journal, timeframe=timeframe)

# %%
print(account)
print(journal)
