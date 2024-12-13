"""This example shows how to perform a walk-forward using a multi-process approach.
Each run is over a certain timeframe and set of parameters for the EMA Crossover strategy.
"""

from multiprocessing import get_context
from itertools import product

import roboquant as rq

# Feed with over 20 years of data
FEED = rq.feeds.YahooFeed("GOOG", "MSFT", "NVDA", start_date="2000-01-01")
print(FEED)


def _walkforward(params):
    """Perform a run over the provided timeframe"""
    timeframe, (fast, slow) = params 
    strategy = rq.strategies.EMACrossover(fast, slow)
    acc = rq.run(FEED, strategy, timeframe=timeframe)
    print(f"{timeframe} EMA({fast:2},{slow:2}) ==> {acc.equity()}")
    return acc.equity_value()

if __name__ == "__main__":

    # Using "fork" ensures that the FEED object is not being created for each process
    # The pool is created with default number of processes (equals CPU cores available) 
    with get_context("fork").Pool() as p:
        # Split overal timeframe into 5 equal non-overlapping timeframes
        timeframes = FEED.timeframe().split(5)

        # Test the following combinations of parameters for EMACrossover strategy
        ema_params = [(3, 5), (5, 7), (10, 15), (15, 21)]
        params = product(timeframes, ema_params)

        # run the back tests in parallel
        equities = p.map(_walkforward, params)

        # print some result
        print("max equity =>", max(equities))
        print("min equity =>", min(equities))

