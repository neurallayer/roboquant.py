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

    # Connect to Tiingo and subscribe to some stocks
    src_feed = rq.feeds.TiingoLiveFeed(market="iex")
    src_feed.subscribe("TSLA", "AMZN", "MSFT", "META", "IBM")

    # Convert the trades into 15-second candles
    feed = rq.feeds.CandleFeed(src_feed, timedelta(seconds=15))

    # Lets run our EMACrossover strategy for 15 minutes
    roboquant = rq.Roboquant(rq.strategies.EMACrossover(3, 5), broker=ibkr)
    timeframe = rq.Timeframe.next(minutes=15)
    tracker = rq.trackers.BasicTracker()
    account = roboquant.run(feed, tracker, timeframe)
    src_feed.close()

    print(account)
    print(tracker)
    sys.exit(0)
