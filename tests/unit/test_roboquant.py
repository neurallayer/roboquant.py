import unittest
from datetime import timedelta
import roboquant as rq
from tests.common import get_feed


class TestRoboquant(unittest.TestCase):

    def setUp(self):
        self.feed = get_feed()

    def test_single_run(self):
        roboquant = rq.Roboquant(rq.strategies.EMACrossover())
        account = roboquant.run(self.feed)
        self.assertEqual(self.feed.timeframe().end, account.last_update)

    def test_walkforward_run(self):
        account = None
        for tf in self.feed.timeframe().split(5):
            roboquant = rq.Roboquant(rq.strategies.EMACrossover())
            account = roboquant.run(self.feed, timeframe=tf)
            self.assertLessEqual(account.last_update, tf.end)

        if account:
            self.assertEqual(self.feed.timeframe().end, account.last_update)
        else:
            self.fail()

    def test_montecarlo_run(self):
        for tf in self.feed.timeframe().sample(timedelta(days=265), 10):
            roboquant = rq.Roboquant(rq.strategies.EMACrossover())
            account = roboquant.run(self.feed, timeframe=tf)
            self.assertLessEqual(account.last_update, tf.end)


if __name__ == "__main__":
    unittest.main()
