"""This example shows how to perform a walk-forward using a multi-process approach.
This allows you to utilize all the CPU's available on the machine (at the cost of higher memory usage)

Each run is over a certain timeframe and set of parameters for the EMA Crossover strategy.
"""

from multiprocessing import get_context
from itertools import product

import roboquant as rq

# Feed with over 25 years of data
FEED = rq.feeds.YahooFeed("GOOG", "MSFT", "NVDA", start_date="2000-01-01")
print(FEED)


def walk_forward(params):
    """Perform a run over the provided timeframe and EMA parameters"""
    timeframe, (fast, slow) = params
    strategy = rq.strategies.EMACrossover(fast, slow)
    acc = rq.run(FEED, strategy, timeframe=timeframe)
    print(f"{timeframe} EMA({fast:2},{slow:2}) ==> {acc.equity()}")
    return acc.equity_value()


if __name__ == "__main__":

    # Using "fork" ensures that the FEED object is not being recreated for each process
    # The pool is created with default number of processes (equal to the number of CPU cores)
    with get_context("fork").Pool() as p:

        # Split overal timeframe into 5 equal non-overlapping timeframes
        timeframe_params = FEED.timeframe().split(5)

        # EMACrossover parameters, the fast and slow periods
        ema_params = [(3, 5), (5, 7), (10, 15), (15, 21)]

        # All the combinations of parameters (Cartesian product)
        all_params = product(timeframe_params, ema_params)

        # run the walk-forward in parallel
        equities = p.map(walk_forward, all_params)

        # print some result
        print("max equity =>", max(equities))
        print("min equity =>", min(equities))
