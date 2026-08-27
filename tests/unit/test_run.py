import unittest
from datetime import timedelta
from typing import override

import roboquant as rq
import math
from tests.common import get_feed


class TestRoboquant(unittest.TestCase):

    @override
    def setUp(self):
        self.feed = get_feed()

    def test_single_run(self):
        journal = rq.journals.BasicJournal()
        account = rq.run(self.feed, rq.strategies.EMACrossover(), journal=journal)
        self.assertEqual(self.feed.timeframe().end, account.last_update)
        self.assertEqual(self.feed.count_items(), journal.items)

    def test_walkforward_run(self):
        account = None
        for tf in self.feed.timeframe().split(5):
            account = rq.run(self.feed, rq.strategies.EMACrossover(), timeframe=tf)
            self.assertLessEqual(account.last_update, tf.end)
            for p in account.portfolio:
                self.assertTrue(math.isfinite(p.mkt_price))
                self.assertTrue(math.isfinite(p.avg_price))
                self.assertTrue(p.size != 0)

        if account:
            self.assertEqual(self.feed.timeframe().end, account.last_update)
        else:
            self.fail()

    def test_montecarlo_run(self):
        for tf in self.feed.timeframe().sample(10, timedelta(days=265)):
            account = rq.run(self.feed, rq.strategies.EMACrossover(), timeframe=tf)
            self.assertLessEqual(account.last_update, tf.end)


if __name__ == "__main__":
    unittest.main()
