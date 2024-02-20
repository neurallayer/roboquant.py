import unittest
import talib.stream as ta
from roboquant import CandleStrategy, OHLCVBuffer
from tests.common import run_strategy


class MyTaLibStrategy(CandleStrategy):
    """Example using talib to create a strategy"""

    def _give_rating(self, _, ohlcv: OHLCVBuffer) -> float | None:
        close = ohlcv.close()
        ema12 = ta.EMA(close, 12)  # type: ignore
        ema26 = ta.EMA(close, 26)  # type: ignore
        if ema12 > ema26:
            return 1.0
        if ema12 < ema26:
            return -1.0


class TestOHLCVStrategy(unittest.TestCase):

    def test_ohlcv_strategy(self):
        # ensure there is enough history available
        # for the used talib indicators
        strategy = MyTaLibStrategy(27)
        run_strategy(strategy, self)


if __name__ == "__main__":
    unittest.main()
