from datetime import timedelta
import logging
from roboquant import Roboquant, EMACrossover, Timeframe, BasicTracker, TiingoLiveFeed, CandleFeed
from roboquant.brokers.ibkrbroker import IBKRBroker
import sys

if __name__ == "__main__":
    logging.basicConfig()
    logging.getLogger("roboquant").setLevel(level=logging.INFO)

    # Connect to local running TWS or IB Gateway
    ibkr = IBKRBroker()

    # Connect to Tiingo and subscribe to some stocks
    src_feed = TiingoLiveFeed(market="iex")
    src_feed.subscribe("TSLA", "AMZN", "MSFT", "META", "IBM")

    # Convert the trades into 15-second candles
    feed = CandleFeed(src_feed, timedelta(seconds=15))

    # Lets run our EMACrossover strategy for 15 minutes
    rq = Roboquant(EMACrossover(3, 5), broker=ibkr)
    timeframe = Timeframe.next(minutes=15)
    tracker = BasicTracker()
    account = rq.run(feed, tracker, timeframe)
    src_feed.close()

    print(account)
    print(tracker)
    sys.exit(0)
