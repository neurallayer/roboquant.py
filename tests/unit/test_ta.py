import unittest

from roboquant.signal import Signal
from roboquant.strategies.buffer import OHLCVBuffer
from roboquant.strategies.tastrategy import TaStrategy
import roboquant.ta as ta
import numpy as np

from tests.common import run_strategy

class _MyStrategy(TaStrategy):
    """Example using TaStrategy as a baseclass to create a custom strategy
    using talib for the technical indicators.
    """

    def process_asset(self, asset, ohlcv: OHLCVBuffer):
        close = ohlcv.close()
        sma12 = ta.SMA(close, timeperiod=12)
        sma26 = ta.SMA(close, timeperiod=26)

        if sma12 > sma26:
            return Signal.buy(asset)
        if sma12 < sma26:
            return Signal.sell(asset)
        return None


class TestTa(unittest.TestCase):

    def test_ta_indicator(self):
        # ensure there is enough history available
        data = [i for i in range(1, 100)]
        np_data = np.array(data, dtype=ta.np.float64)
        sma = ta.SMA(np_data, timeperiod=5)
        self.assertIsInstance(sma, float)
        self.assertAlmostEqual(sma, 97.0, places=4)

    def test_my_ta_strategy(self):
        # ensure there is enough history available
        strategy = _MyStrategy(27)
        run_strategy(strategy, self)

if __name__ == "__main__":
    unittest.main()
