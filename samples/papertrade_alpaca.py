# %%
from datetime import timedelta
import logging
import roboquant as rq
from roboquant.brokers.alpacabroker import AlpacaBroker

# %%
logging.basicConfig()
logging.getLogger("roboquant").setLevel(level=logging.INFO)

# %%
broker = AlpacaBroker()
account = broker.sync()
print(account)

# %%
# Connect to Alpaca and subscribe to popular S&P-500 stocks
alpaca_feed = rq.feeds.AlpacaLiveFeed(market="iex")
symbols = ["TSLA", "MSFT", "NVDA", "AMD", "AAPL", "AMZN", "META", "GOOG", "XOM", "JPM", "NLFX", "BA", "INTC", "V"]
alpaca_feed.subscribe_trades(*symbols)

# Convert the trades into 15-second candles
feed = rq.feeds.AggregatorFeed(alpaca_feed, timedelta(seconds=15))

# %%
strategy = rq.strategies.EMACrossover(13, 26)
timeframe = rq.Timeframe.next(minutes=15)
journal = rq.journals.BasicJournal()
account = rq.run(feed, strategy, broker=broker, journal=journal, timeframe=timeframe)

# %%
print(account)
print(journal)


# %%
