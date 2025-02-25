import os
import unittest
from cProfile import Profile
from pstats import Stats, SortKey

import roboquant as rq


class TestProfile(unittest.TestCase):
    """Collect profiling statistics over a simple backtest. This can be used to detect
    performance bottlenecks and to optimize the code."""

    def test_profile(self):
        path = os.path.expanduser("~/data/nasdaq_stocks/1")
        feed = rq.feeds.CSVFeed.stooq_us_daily(path)
        print(feed)
        strategy = rq.strategies.EMACrossover()
        journal = rq.journals.BasicJournal()

        # Profile the run to detect bottlenecks
        with Profile() as profile:
            rq.run(feed, strategy, journal=journal)
            print(f"\n{journal}")
            Stats(profile).sort_stats(SortKey.TIME).print_stats()


if __name__ == "__main__":
    unittest.main()
