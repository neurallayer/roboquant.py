import unittest

from roboquant.strategies import EMACrossover, MultiStrategy, IBSStrategy, CachedStrategy
from tests.common import run_strategy, get_feed


class TestStrategy(unittest.TestCase):

    def test_ibs_strategy(self):
        strategy = IBSStrategy()
        run_strategy(strategy, self)

    def test_ema_strategy(self):
        strategy = EMACrossover(13, 26)
        run_strategy(strategy, self)

    def test_multi_strategies(self):
        strategy = MultiStrategy(
            EMACrossover(13, 26),
            EMACrossover(5, 12),
            EMACrossover(2, 10),
        )
        run_strategy(strategy, self)

    def test_caching_strategy(self):
        strategy = EMACrossover()
        feed = get_feed()
        caching_strategy = CachedStrategy(feed, strategy)
        self.assertEqual(feed.timeframe(), caching_strategy.timeframe())
        run_strategy(caching_strategy, self)

if __name__ == "__main__":
    unittest.main()
