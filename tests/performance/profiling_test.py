from cProfile import Profile
import os
from pstats import Stats, SortKey
import roboquant as rq


if __name__ == "__main__":
    path = os.path.expanduser("~/data/nasdaq_stocks/1")
    feed = rq.feeds.CSVFeed.stooq_us_daily(path)
    print("timeframe =", feed.timeframe(), " symbols =", len(feed.symbols))
    roboquant = rq.Roboquant(rq.strategies.EMACrossover(13, 26))
    journal = rq.journals.BasicJournal()

    # Profile the run to detect bottlenecks
    with Profile() as profile:
        roboquant.run(feed, journal)
        print(f"\n{journal}")
        Stats(profile).sort_stats(SortKey.TIME).print_stats()
