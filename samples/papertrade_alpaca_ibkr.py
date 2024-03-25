# %%
from datetime import timedelta
import logging
import roboquant as rq
from roboquant.account import Account, CurrencyConverter
from roboquant.brokers.ibkr import IBKRBroker

# %%
logging.basicConfig()
logging.getLogger("roboquant").setLevel(level=logging.INFO)

# %%
# Connect to local running TWS or IB Gateway
converter = CurrencyConverter("EUR", "USD")
converter.register_rate("USD", 0.91)
Account.register_converter(converter)
ibkr = IBKRBroker.use_tws()

# %%
# Connect to Alpaca and subscribe to popular S&P-500 stocks
alpaca_feed = rq.feeds.AlpacaLiveFeed(market="iex")
symbols = ["TSLA", "MSFT", "NVDA", "AMD", "AAPL", "AMZN", "META", "GOOG", "XOM", "JPM", "NLFX", "BA", "INTC", "V"]
alpaca_feed.subscribe_trades(*symbols)

# Convert the trades into 15-second candles
feed = rq.feeds.AggregatorFeed(alpaca_feed, timedelta(seconds=15))

# %%
strategy = rq.strategies.EMACrossover(13, 26)
timeframe = rq.Timeframe.next(minutes=60)
journal = rq.journals.BasicJournal()
account = rq.run(feed, strategy, broker=ibkr, journal=journal, timeframe=timeframe)
ibkr.disconnect()

# %%
print(account)
print(journal)

