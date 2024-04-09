# %%
from datetime import timedelta
import logging
import random
import roboquant as rq
from roboquant.account import Account, CurrencyConverter
from roboquant.brokers.ibkr import IBKRBroker

from roboquant.feeds.feedutil import get_sp500_symbols

# %%
logging.basicConfig()
logging.getLogger("roboquant").setLevel(level=logging.INFO)

# %%
# Connect to local running TWS
converter = CurrencyConverter("EUR", "USD")
converter.register_rate("USD", 0.91)
Account.register_converter(converter)
ibkr = IBKRBroker.use_tws()

# %%
# Connect to Tiingo and subscribe to 10 S&P-500 stocks
tiingo_feed = rq.feeds.TiingoLiveFeed(market="iex")
symbols = random.sample(get_sp500_symbols(), 10)
tiingo_feed.subscribe(*symbols)

# Convert the trades into 15-second candles
feed = rq.feeds.AggregatorFeed(tiingo_feed, timedelta(seconds=15))

# %%
strategy = rq.strategies.EMACrossover(13, 26)
timeframe = rq.Timeframe.next(minutes=15)
journal = rq.journals.BasicJournal()
account = rq.run(feed, strategy, broker=ibkr, journal=journal, timeframe=timeframe)
tiingo_feed.close()
ibkr.disconnect()

# %%
print(account)
print(journal)

