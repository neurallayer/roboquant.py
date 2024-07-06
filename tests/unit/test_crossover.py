import unittest

from roboquant.strategies import EMACrossover
from tests.common import run_strategy


class TestCrossover(unittest.TestCase):

    def test_ema_crossover(self):
        strategy = EMACrossover(13, 26)
        run_strategy(strategy, self)


if __name__ == "__main__":
    unittest.main()
