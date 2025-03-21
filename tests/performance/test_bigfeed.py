import os
import time
import unittest

import roboquant as rq


class TestBigFeed(unittest.TestCase):
    """Run two large back tests, one over daily bars and one over 5-minutes bars"""

    @staticmethod
    def _print(account, journal: rq.journals.BasicJournal, n_assets, load_time, runtime):
        print("", account, journal, sep="\n\n")

        candles = journal.items / 1_000_000.0
        throughput = candles / runtime

        # Print statistics
        print()
        print(f"load time  = {load_time:.1f}s")
        print(f"files      = {n_assets}")
        print(f"throughput = {n_assets / load_time:.0f} files/s")
        print(f"run time   = {runtime:.1f}s")
        print(f"candles    = {candles:.1f}M")
        print(f"throughput = {throughput:.1f}M candles/s")
        print()

    def _run(self, feed, journal: rq.journals.BasicJournal):
        strategy = rq.strategies.EMACrossover(13, 26)
        start = time.time()
        account = rq.run(feed, strategy, journal=journal)
        self.assertTrue(journal.items > 1_000_000)
        self.assertTrue(journal.events > 1_000)
        return account, time.time() - start

    def test_big_feed_daily(self):
        print("============ Daily Bars ============")
        start = time.time()
        path = os.path.expanduser("~/data/nyse_stocks/")
        feed = rq.feeds.CSVFeed.stooq_us_daily(path)
        load_time = time.time() - start

        journal = rq.journals.BasicJournal()
        account, runtime = self._run(feed, journal)
        self._print(account, journal, len(feed.assets()), load_time, runtime)

    def test_big_feed_intraday(self):
        print("============ 5 Min Bars ============")
        start = time.time()
        path = os.path.expanduser("~/data/intra/")
        feed = rq.feeds.CSVFeed.stooq_us_intraday(path)
        load_time = time.time() - start

        journal = rq.journals.BasicJournal()
        account, runtime = self._run(feed, journal)
        self._print(account, journal, len(feed.assets()), load_time, runtime)


if __name__ == "__main__":
    unittest.main()
