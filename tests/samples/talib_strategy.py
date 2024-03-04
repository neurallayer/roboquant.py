import unittest
import talib.stream as ta
from roboquant import BUY, SELL, Signal
from roboquant.strategies import CandleStrategy
from tests.common import run_strategy


class EMAStrategy(CandleStrategy):
    """Example using talib to create an EMA crossover strategy"""

    def _create_signal(self, symbol, ohlcv) -> Signal | None:
        close = ohlcv.close()
        ema12 = ta.EMA(close, 12)  # type: ignore pylint: disable=no-member
        ema26 = ta.EMA(close, 26)  # type: ignore pylint: disable=no-member
        if ema12 > ema26:
            return BUY
        if ema12 < ema26:
            return SELL
        return None


class RSIStrategy(CandleStrategy):
    """Example using talib to create an RSI strategy"""

    def __init__(self, period):
        super().__init__(period + 1)

    def _create_signal(self, symbol, ohlcv) -> Signal | None:
        close = ohlcv.close()
        rsi = ta.RSI(close, self.size - 1)  # type: ignore pylint: disable=no-member
        if rsi < 30:
            return BUY
        if rsi > 70:
            return SELL
        return None


class TestOHLCVStrategy(unittest.TestCase):

    def test_ohlcv_strategy(self):
        # ensure there is enough history available
        # for the used talib indicators
        strategy = EMAStrategy(27)
        run_strategy(strategy, self)

        strategy = RSIStrategy(14)
        run_strategy(strategy, self)


if __name__ == "__main__":
    unittest.main()
