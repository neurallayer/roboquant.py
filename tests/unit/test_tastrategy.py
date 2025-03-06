import unittest

from roboquant.signal import Signal
from roboquant.strategies import TaStrategy, OHLCVBuffer
from tests.common import run_strategy


class _MyStrategy(TaStrategy):
    """Example using TaStrategy as a baseclass to create a custom strategy"""

    def process_asset(self, asset, ohlcv: OHLCVBuffer):
        close = ohlcv.close()
        sma12 = close[-12:].mean()
        sma26 = close[-26:].mean()
        if sma12 > sma26:
            return Signal.buy(asset)
        if sma12 < sma26:
            return Signal.sell(asset)
        return None


class TestTaStrategy(unittest.TestCase):

    def test_my_tastrategy(self):
        # ensure there is enough history available
        strategy = _MyStrategy(27)
        run_strategy(strategy, self)


if __name__ == "__main__":
    unittest.main()
