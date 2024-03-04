import unittest

from roboquant.strategies import EMACrossover, MultiStrategy
from tests.common import run_strategy


class TestMultiStrategy(unittest.TestCase):

    def test_multi_strategies(self):
        strategy = MultiStrategy(
            EMACrossover(13, 26),
            EMACrossover(5, 12),
            EMACrossover(2, 10),
        )
        run_strategy(strategy, self)


if __name__ == "__main__":
    unittest.main()
