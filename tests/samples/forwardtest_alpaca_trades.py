# %%
import os
from datetime import timedelta
import logging
import roboquant as rq
from roboquant.alpaca import AlpacaLiveFeed
from dotenv import load_dotenv
load_dotenv()

# %%
logging.basicConfig()
logging.getLogger("roboquant").setLevel(level=logging.INFO)

# Connect to Alpaca and subscribe to some IEX stocks
symbols = ["TSLA", "MSFT", "NVDA", "AMD", "AAPL", "AMZN", "META", "GOOG", "XOM", "JPM", "NLFX", "BA", "INTC", "V"]
api_key = os.environ["ALPACA_API_KEY"]
secret_key = os.environ["ALPACA_SECRET"]
alpaca = AlpacaLiveFeed(api_key, secret_key, market="iex")
alpaca.subscribe_trades(*symbols)

# Convert the trades into 15-second bars
feed = rq.feeds.BarAggregatorFeed(alpaca, timedelta(seconds=15), "trade")

# %%
# Let run an EMACrossover strategy
strategy = rq.strategies.EMACrossover(13, 26)
timeframe = rq.Timeframe.next(minutes=30)
journal = rq.journals.BasicJournal()
account = rq.run(feed, strategy, journal=journal, timeframe=timeframe)

# %%
print(account)
print(journal)
