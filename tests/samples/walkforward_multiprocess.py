from multiprocessing import get_context

import roboquant as rq

# Feed with over 20 years of data
FEED = rq.feeds.YahooFeed("GOOG", "MSFT", "NVDA", start_date="2000-01-01")
print(FEED)


def _walkforward(timeframe: rq.Timeframe):
    """Perform a run over the provided timeframe"""
    print("starting:", timeframe)
    strategy = rq.strategies.EMACrossover()
    acc = rq.run(FEED, strategy, timeframe=timeframe)
    print(timeframe, "==>", acc.equity())


if __name__ == "__main__":

    # Using "fork" ensures that the FEED object is not being created for each process
    # The pool is created with default number of processes (equals CPU cores available) 
    with get_context("fork").Pool() as p:
        # Perform a walkforward over 8 equal timeframes
        timeframes = FEED.timeframe().split(8)
        p.map(_walkforward, timeframes)

