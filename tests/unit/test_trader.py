from typing import override
import unittest

from roboquant.common.account import Account
from roboquant.common.monetary import Amount, USD, Wallet
from roboquant.common.signal import Signal
from roboquant.traders.flextrader import FlexTrader
from roboquant.traders.simpletrader import SimpleTrader
from tests.common import get_feed


class TestTrader(unittest.TestCase):

    @override
    def setUp(self) -> None:
        acc = Account.with_defaults(
            buying_power=Amount(USD, 100_000.0),
            cash=Wallet(Amount(USD, 100_000.0)),
        )
        self.account = acc
        self.feed = get_feed()

    def test_flex_trader(self):
        trader = FlexTrader()
        for evt in self.feed.play():
            orders = trader.create_orders([], evt, self.account)
            self.assertFalse(orders)

            asset = next(iter(evt.get_prices().keys()))
            signals = [Signal.buy(asset)]
            orders = trader.create_orders(signals, evt, self.account)
            self.assertEqual(len(signals), len(orders))
            self.assertEqual(signals[0].asset, orders[0].asset)
            self.assertEqual(signals[0].is_buy, orders[0].is_buy)

    def test_fixed_trader(self):
        trader = SimpleTrader()
        for evt in self.feed.play():
            asset = next(iter(evt.get_prices().keys()))
            signals = [Signal.buy(asset)]
            orders = trader.create_orders(signals, evt, self.account)
            self.assertEqual(len(signals), len(orders))
            self.assertEqual(signals[0].asset, orders[0].asset)
            self.assertEqual(signals[0].is_buy, orders[0].is_buy)


if __name__ == "__main__":
    unittest.main()
