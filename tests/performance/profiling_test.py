from cProfile import Profile
import os
from pstats import Stats, SortKey

from roboquant import Roboquant, CSVFeed, BasicTracker, EMACrossover

if __name__ == "__main__":
    path = os.path.expanduser("~/data/nasdaq_stocks/1")
    feed = CSVFeed.stooq_us_daily(path)
    print("timeframe =", feed.timeframe(), " symbols =", len(feed.symbols))
    rq = Roboquant(EMACrossover(13, 26))
    tracker = BasicTracker()

    # Profile the run to detect bottlenecks
    with Profile() as profile:
        rq.run(feed, tracker=tracker)
        print(tracker)
        Stats(profile).sort_stats(SortKey.TIME).print_stats()
