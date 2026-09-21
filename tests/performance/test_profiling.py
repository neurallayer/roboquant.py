import os
from cProfile import Profile
from pstats import Stats, SortKey

import roboquant as rq

print("Loading large set of CSV files ...")
path = os.path.expanduser("~/data/daily/us/nasdaq stocks/1")
feed = rq.feeds.CSVFeed.stooq_us_daily(path)
print(f"timeframe: {feed.timeframe()}")
print(f"number of assets: {len(feed.assets())}")

def test_profile():
    print("\n\nRegular strategy\n##################################")
    strategy = rq.strategies.EMACrossover()
    journal = rq.journals.BasicJournal()

    # Profile the run to detect bottlenecks
    with Profile() as profile:
        rq.run(feed, strategy, journal=journal)
        print(f"\n{journal}")
        Stats(profile).sort_stats(SortKey.TIME).print_stats(.1, "roboquant")

if __name__ == "__main__":
    test_profile()
