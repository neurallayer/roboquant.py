import os
import unittest
from cProfile import Profile
from pstats import Stats, SortKey

import roboquant as rq


path = os.path.expanduser("~/data/daily/us/nasdaq stocks/1")
feed = rq.feeds.CSVFeed.stooq_us_daily(path)

class TestProfile(unittest.TestCase):
    """Collect profiling statistics over a simple backtest. This can be used to detect
    performance bottlenecks and to optimize the code."""

    def test_profile_1(self):
        print("\n\nRegular stratetgy\n##################################")
        strategy = rq.strategies.EMACrossover()
        journal = rq.journals.BasicJournal()

        # Profile the run to detect bottlenecks
        with Profile() as profile:
            rq.run(feed, strategy, journal=journal)
            print(f"\n{journal}")
            Stats(profile).sort_stats(SortKey.TIME).print_stats(.1, "roboquant")


    def test_profile_2(self):
        print("\n\nCached stratetgy\n##################################")

        strategy = rq.strategies.EMACrossover()
        cached_strategy = rq.strategies.CachedStrategy(feed, strategy)
        journal = rq.journals.BasicJournal()

        # Profile the run to detect bottlenecks
        with Profile() as profile:
            rq.run(feed, cached_strategy, journal=journal)
            print(f"\n{journal}")
            Stats(profile).sort_stats(SortKey.TIME).print_stats(.1, "roboquant")


if __name__ == "__main__":
    unittest.main()
