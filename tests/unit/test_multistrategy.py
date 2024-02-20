import unittest
from roboquant import EMACrossover
from roboquant.strategies.multistrategy import MultiStrategy
from tests.common import run_strategy


class TestMultiStrategy(unittest.TestCase):

    def test_multistrategies(self):
        strategy = MultiStrategy(
            EMACrossover(13, 26),
            EMACrossover(5, 12),
            EMACrossover(2, 10),
        )
        run_strategy(strategy, self)


if __name__ == "__main__":
    unittest.main()
