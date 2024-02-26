import os
import time
import unittest
import roboquant as rq


class TestBigFeed(unittest.TestCase):

    def test_bigfeed(self):
        start = time.time()
        path = os.path.expanduser("~/data/nyse_stocks/")
        feed = rq.feeds.CSVFeed.stooq_us_daily(path)
        loadtime = time.time() - start
        strategy = rq.strategies.EMACrossover(13, 26)
        journal = rq.journals.BasicJournal()
        start = time.time()
        account = rq.run(feed, strategy, journal=journal)
        runtime = time.time() - start

        self.assertTrue(journal.items > 1_000_000)
        self.assertTrue(journal.signals > 100_000)
        self.assertTrue(journal.orders > 10_000)
        self.assertTrue(journal.events > 10_000)

        print(account)
        print(journal)

        # Print statistics
        print()
        print(f"load time  = {loadtime:.1f}s")
        print("files      =", len(feed.symbols))
        print(f"throughput = {len(feed.symbols) / loadtime:.0f} files/s")
        print(f"run time   = {runtime:.1f}s")
        candles = journal.items
        print(f"candles    = {(candles / 1_000_000):.1f}M")
        throughput = candles / (runtime * 1_000_000)
        print(f"throughput = {throughput:.1f}M candles/s")
        print()


if __name__ == "__main__":
    unittest.main()
