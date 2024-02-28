from datetime import timedelta
import logging
import roboquant as rq
from roboquant.brokers.ibkrbroker import IBKRBroker
import sys

if __name__ == "__main__":
    logging.basicConfig()
    logging.getLogger("roboquant").setLevel(level=logging.INFO)

    # Connect to local running TWS or IB Gateway
    ibkr = IBKRBroker()

    # Connect to Tiingo and subscribe to all stocks
    src_feed = rq.feeds.TiingoLiveFeed(market="iex")
    src_feed.subscribe()

    # Convert the trades into 15-second candles
    feed = rq.feeds.CandleFeed(src_feed, timedelta(seconds=60))

    # Let run our EMACrossover strategy for 10 minutes
    strategy = rq.strategies.EMACrossover(3, 5)
    timeframe = rq.Timeframe.next(minutes=10)
    journal = rq.journals.BasicJournal()
    account = rq.run(feed, strategy, broker=ibkr, journal=journal, timeframe=timeframe)
    src_feed.close()

    print(account)
    print(journal)
    sys.exit(0)
