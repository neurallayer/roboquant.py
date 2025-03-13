# %%
import os
from datetime import timedelta
import logging
import roboquant as rq
from roboquant.alpaca import AlpacaBroker, AlpacaLiveFeed
from dotenv import load_dotenv
load_dotenv()

# %%
logging.basicConfig()
logging.getLogger("roboquant").setLevel(level=logging.INFO)

# %%
api_key =  os.environ["ALPACA_API_KEY"]
secret_key = os.environ["ALPACA_SECRET"]
broker = AlpacaBroker(api_key, secret_key)
account = broker.sync()
print(account)

# %%
# Connect to Alpaca and subscribe to some popular stocks
alpaca_feed = AlpacaLiveFeed(api_key, secret_key, market="iex")
symbols = ["TSLA", "MSFT", "NVDA", "AMD", "AAPL", "AMZN", "META", "GOOG", "XOM", "JPM", "NLFX", "BA", "INTC", "V"]
alpaca_feed.subscribe_trades(*symbols)

# Convert the trades into 15-second candles
feed = rq.feeds.BarAggregatorFeed(alpaca_feed, timedelta(seconds=15), price_type="trade")

# %%
strategy = rq.strategies.EMACrossover(13, 26)
timeframe = rq.Timeframe.next(minutes=15)
journal = rq.journals.BasicJournal()
account = rq.run(feed, strategy, broker=broker, journal=journal, timeframe=timeframe)

# %%
print(account)
print(journal)
