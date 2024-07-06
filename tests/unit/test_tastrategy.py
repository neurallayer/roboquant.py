import unittest

from roboquant.event import Bar
from roboquant.strategies import TaStrategy, OHLCVBuffer
from roboquant.strategies.buffer import OHLCVBuffers
from roboquant.strategies.ordermanager import FlexOrderManager
from roboquant.strategies.strategy import Strategy
from tests.common import run_strategy


class _MyStrategy(TaStrategy):
    """Example using CandleStrategy to create a custom strategy"""

    def process_symbol(self, symbol, ohlcv: OHLCVBuffer, item: Bar):
        close = ohlcv.close()
        sma12 = close[-12:].mean()
        sma26 = close[-26:].mean()
        if sma12 > sma26:
            self.add_buy_order(symbol)
        if sma12 < sma26:
            self.add_sell_order(symbol)


class _MyStrategy2(Strategy):
    """Example using CandleStrategy to create a custom strategy"""

    def __init__(self):
        self.data = OHLCVBuffers(10)
        self.om = FlexOrderManager()

    def create_orders(self, event, account):
        self.om.next(event, account)

        for symbol in self.data.add_event(event):
            close = self.data[symbol].close()
            sma12 = close[-12:].mean()
            sma26 = close[-26:].mean()
            if sma12 > sma26:
                self.om.add_buy_order(symbol)
            if sma12 < sma26:
                self.om.add_sell_order(symbol)

        return self.om.get_orders()


class TestCandleStrategy(unittest.TestCase):

    def test_candle_strategy(self):
        # ensure there is enough history available
        strategy = _MyStrategy(27)
        run_strategy(strategy, self)


if __name__ == "__main__":
    unittest.main()
