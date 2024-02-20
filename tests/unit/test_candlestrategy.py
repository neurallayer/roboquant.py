import unittest
from roboquant import CandleStrategy, OHLCVBuffer
from tests.common import run_strategy


class _MyStrategy(CandleStrategy):
    """Example using talib to create a strategy"""

    def _give_rating(self, _, ohlcv: OHLCVBuffer) -> float | None:
        close = ohlcv.close()
        sma12 = close[-12:].mean()
        sma26 = close[-26:].mean()  # type: ignore
        if sma12 > sma26:
            return 1.0
        if sma12 < sma26:
            return -1.0


class TestCandleStrategy(unittest.TestCase):

    def test_candle_strategy(self):
        # ensure there is enough history available
        strategy = _MyStrategy(27)

        run_strategy(strategy, self)


if __name__ == "__main__":
    unittest.main()
