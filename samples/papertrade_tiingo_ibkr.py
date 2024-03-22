from datetime import timedelta
import logging
import roboquant as rq
from roboquant.account import Account, CurrencyConverter
from roboquant.brokers.ibkr import IBKRBroker

from roboquant.feeds.feedutil import get_sp500_symbols

if __name__ == "__main__":
    logging.basicConfig()
    logging.getLogger("roboquant").setLevel(level=logging.INFO)

    # Connect to local running TWS or IB Gateway
    converter = CurrencyConverter("EUR", "USD")
    converter.register_rate("USD", 0.91)
    Account.register_converter(converter)
    ibkr = IBKRBroker.use_tws()

    # Connect to Tiingo and subscribe to S&P-500 stocks
    src_feed = rq.feeds.TiingoLiveFeed(market="iex")
    sp500 = get_sp500_symbols()
    src_feed.subscribe(*sp500)

    # Convert the trades into 15-second candles
    feed = rq.feeds.AggregatorFeed(src_feed, timedelta(seconds=15))

    # Let run an EMACrossover strategy
    strategy = rq.strategies.EMACrossover(13, 26)
    timeframe = rq.Timeframe.next(minutes=10)
    journal = rq.journals.BasicJournal()
    account = rq.run(feed, strategy, broker=ibkr, journal=journal, timeframe=timeframe)
    src_feed.close()

    print(account)
    print(journal)
    ibkr.disconnect()
