import unittest

from roboquant.strategies.signal import Signal
from roboquant.strategies import TaStrategy, OHLCVBuffer
from tests.common import run_strategy


class _MyStrategy(TaStrategy):
    """Example using CandleStrategy to create a custom strategy"""

    def _create_signal(self, symbol, ohlcv: OHLCVBuffer) -> Signal | None:
        close = ohlcv.close()
        sma12 = close[-12:].mean()
        sma26 = close[-26:].mean()
        if sma12 > sma26:
            return Signal.buy(symbol)
        if sma12 < sma26:
            return Signal.sell(symbol)
        return None


class TestCandleStrategy(unittest.TestCase):

    def test_candle_strategy(self):
        # ensure there is enough history available
        strategy = _MyStrategy(27)

        run_strategy(strategy, self)


if __name__ == "__main__":
    unittest.main()
