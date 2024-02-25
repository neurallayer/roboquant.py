import unittest
from roboquant.strategies import SMACrossover
from tests.common import run_strategy


class TestCrossover(unittest.TestCase):

    def test_smacrossover(self):
        strategy = SMACrossover(13, 26)
        run_strategy(strategy, self)

    def test_emacrossover(self):
        strategy = SMACrossover(13, 26)
        run_strategy(strategy, self)


if __name__ == "__main__":
    unittest.main()
