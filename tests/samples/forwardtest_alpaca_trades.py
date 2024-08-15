# %%
from datetime import timedelta
import logging
import roboquant as rq
from roboquant.alpaca import AlpacaLiveFeed

# %%
logging.basicConfig()
logging.getLogger("roboquant").setLevel(level=logging.INFO)

# Connect to Alpaca and subscribe to some IEX stocks
symbols = ["TSLA", "MSFT", "NVDA", "AMD", "AAPL", "AMZN", "META", "GOOG", "XOM", "JPM", "NLFX", "BA", "INTC", "V"]
alpaca = AlpacaLiveFeed(market="iex")
alpaca.subscribe_trades(*symbols)

# Convert the trades into 15-second bars
feed = rq.feeds.AggregatorFeed(alpaca, timedelta(seconds=15), "trade")

# %%
# Let run an EMACrossover strategy
strategy = rq.strategies.EMACrossover(13, 26)
timeframe = rq.Timeframe.next(minutes=30)
journal = rq.journals.BasicJournal()
account = rq.run(feed, strategy, journal=journal, timeframe=timeframe)

# %%
print(account)
print(journal)
