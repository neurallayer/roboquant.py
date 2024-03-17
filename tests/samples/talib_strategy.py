import unittest
from math import isfinite
import talib.stream as ta
from roboquant import BUY, SELL, Signal
from roboquant.strategies import BarStrategy
from tests.common import run_strategy
# pylint: disable=no-member


class EMAStrategy(BarStrategy):
    """Example using talib to create an EMA crossover strategy"""

    def __init__(self, fast=12, slow=26) -> None:
        super().__init__(slow + 1)
        self.fast = fast
        self.slow = slow

    def _create_signal(self, symbol, ohlcv) -> Signal | None:
        close = ohlcv.close()
        ema_fast = ta.EMA(close, self.fast)  # type: ignore
        ema_slow = ta.EMA(close, self.slow)  # type: ignore
        assert isfinite(ema_fast) and isfinite(ema_slow)

        if ema_fast > ema_slow:
            return BUY
        if ema_fast < ema_slow:
            return SELL
        return None


class RSIBBANDStrategy(BarStrategy):
    """Example using talib to create an RSI/BollingerBand strategy"""

    def _create_signal(self, symbol, ohlcv) -> Signal | None:
        close = ohlcv.close()

        rsi = ta.RSI(close, timeperiod=self.size-1)  # type: ignore
        assert isfinite(rsi)

        upper, _, lower = ta.BBANDS(  # type: ignore
            close, timeperiod=self.size-1, nbdevup=2, nbdevdn=2
        )
        assert isfinite(upper) and isfinite(lower)

        price = close[-1]

        if rsi < 30 and price < lower:
            return BUY
        if rsi > 70 and price > upper:
            return SELL

        return None


class TestOHLCVStrategy(unittest.TestCase):

    def test_ohlcv_strategy(self):

        strategy = EMAStrategy()
        run_strategy(strategy, self)

        strategy = RSIBBANDStrategy(14)
        run_strategy(strategy, self)


if __name__ == "__main__":
    unittest.main()
