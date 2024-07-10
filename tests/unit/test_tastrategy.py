import unittest

from roboquant.strategies import TaStrategy, OHLCVBuffer
from roboquant.strategies.buffer import OHLCVBuffers
from roboquant.strategies.basestrategy import BaseStrategy
from tests.common import run_strategy


class _MyStrategy(TaStrategy):
    """Example using CandleStrategy to create a custom strategy"""

    def process_asset(self, asset, ohlcv: OHLCVBuffer):
        close = ohlcv.close()
        sma12 = close[-12:].mean()
        sma26 = close[-26:].mean()
        if sma12 > sma26:
            self.add_buy_order(asset)
        if sma12 < sma26:
            self.add_exit_order(asset)


class _MyStrategy2(BaseStrategy):
    """Example using CandleStrategy to create a custom strategy"""

    def __init__(self):
        super().__init__()
        self.data = OHLCVBuffers(10)

    def process(self, event, account):

        for symbol in self.data.add_event(event):
            close = self.data[symbol].close()
            sma12 = close[-12:].mean()
            sma26 = close[-26:].mean()
            if sma12 > sma26:
                self.add_buy_order(symbol)
            if sma12 < sma26:
                self.add_exit_order(symbol)


class TestCandleStrategy(unittest.TestCase):

    def test_candle_strategy(self):
        # ensure there is enough history available
        strategy = _MyStrategy(27)
        run_strategy(strategy, self)


if __name__ == "__main__":
    unittest.main()
