import os
import time
import unittest

import roboquant as rq


class TestBigFeed(unittest.TestCase):

    def test_big_feed(self):
        start = time.time()
        path = os.path.expanduser("~/data/nyse_stocks/")
        feed = rq.feeds.CSVFeed.stooq_us_daily(path)
        load_time = time.time() - start
        strategy = rq.strategies.EMACrossover(13, 26)
        journal = rq.journals.BasicJournal()
        start = time.time()
        account = rq.run(feed, strategy, journal=journal)
        runtime = time.time() - start

        # self.assertTrue(journal.items > 1_000_000)
        # self.assertTrue(journal.signals > 100_000)
        # self.assertTrue(journal.orders > 10_000)
        # self.assertTrue(journal.events > 10_000)

        print("", account, journal, sep="\n\n")

        # Print statistics
        print()
        print(f"load time  = {load_time:.1f}s")
        print("files      =", len(feed.symbols))
        print(f"throughput = {len(feed.symbols) / load_time:.0f} files/s")
        print(f"run time   = {runtime:.1f}s")
        candles = journal.items / 1_000_000.0
        print(f"candles    = {candles:.1f}M")
        throughput = candles / runtime
        print(f"throughput = {throughput:.1f}M candles/s")
        print()


if __name__ == "__main__":
    unittest.main()
