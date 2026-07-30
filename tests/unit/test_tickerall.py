import unittest
from decimal import Decimal
from types import SimpleNamespace

from roboquant.common.asset import Forex
from roboquant.common.monetary import USD, Currency
from roboquant.common.order import Order
from roboquant.tickerall.tickerall_broker import TickerAllBroker
from roboquant.tickerall.tickerall_feed import _to_asset


class _FakeOrders:
    def __init__(self, pending):
        self._pending = pending
        self.placed: list[dict] = []
        self.modified: list[tuple] = []
        self.cancelled: list[int] = []

    def place(self, account_id, *, type, symbol, side, volume, price=None, **kw):  # noqa: A002 - SDK kw name
        self.placed.append({"type": type, "symbol": symbol, "side": side, "volume": volume, "price": price})
        return SimpleNamespace(ticket=999)

    def list_pending(self, account_id):
        return self._pending

    def modify_pending(self, account_id, ticket, *, price=None, **kw):
        self.modified.append((ticket, price))

    def cancel_pending(self, account_id, ticket, **kw):
        self.cancelled.append(ticket)


class _FakeClient:
    """A minimal stand-in for the `tickerall` SDK client, recording the calls the broker makes."""

    def __init__(self, detail, pending=None):
        self.accounts = SimpleNamespace(get=lambda account_id: detail)
        self.orders = _FakeOrders(pending or [])


def _detail(account=None, positions=None):
    return SimpleNamespace(account=account, positions=positions or [])


def _fin(balance=100.0, currency="USD", free_margin=None):
    return SimpleNamespace(balance=balance, currency=currency, free_margin=free_margin)


def _broker(client: _FakeClient) -> TickerAllBroker:
    broker = TickerAllBroker("dummy-key", "acc-test")
    broker._client = client  # type: ignore[assignment]
    return broker


class TestTickerAll(unittest.TestCase):

    def test_symbol_to_asset(self):
        eur = _to_asset("EURUSDm")
        self.assertIsInstance(eur, Forex)
        self.assertEqual(eur.symbol, "EURUSDm")
        self.assertEqual(eur.currency, USD)
        self.assertEqual(_to_asset("EURGBP").currency, Currency("GBP"))
        self.assertEqual(_to_asset("BTCUSDm").currency, USD)
        self.assertEqual(_to_asset("WEIRD1", Currency("JPY")).currency, Currency("JPY"))

    def test_place_market_order(self):
        client = _FakeClient(_detail(_fin()))
        _broker(client).place_orders([Order(Forex("EURUSDm", USD), Decimal("0.01"), float("nan"))])
        self.assertEqual(len(client.orders.placed), 1)
        self.assertEqual(client.orders.placed[0]["type"], "market")
        self.assertEqual(client.orders.placed[0]["side"], "BUY")
        self.assertEqual(client.orders.placed[0]["volume"], 0.01)
        self.assertIsNone(client.orders.placed[0]["price"])

    def test_place_limit_sell_order(self):
        client = _FakeClient(_detail(_fin()))
        _broker(client).place_orders([Order(Forex("EURUSDm", USD), Decimal("-0.02"), 1.05)])
        self.assertEqual(client.orders.placed[0]["type"], "limit")
        self.assertEqual(client.orders.placed[0]["side"], "SELL")
        self.assertEqual(client.orders.placed[0]["volume"], 0.02)
        self.assertEqual(client.orders.placed[0]["price"], 1.05)

    def test_update_and_cancel(self):
        client = _FakeClient(_detail(_fin()))
        broker = _broker(client)
        resting = Order(Forex("EURUSDm", USD), Decimal("0.01"), 1.05, id="123")
        # an order with an id and a non-zero size is an update; the SDK takes an int ticket
        broker.place_orders([resting.modify(limit=1.02)])
        self.assertEqual(client.orders.modified, [(123, 1.02)])
        # an order with an id and a zero size is a cancellation
        broker.place_orders([resting.cancel()])
        self.assertEqual(client.orders.cancelled, [123])

    def test_balance_zero_is_valid(self):
        # a real account can genuinely hold a 0.0 balance; it must not be rejected as "not warm"
        account = _broker(_FakeClient(_detail(_fin(balance=0.0, free_margin=0.0))))._get_account()
        self.assertEqual(account.buying_power.value, 0.0)
        self.assertEqual(account.cash[USD], 0.0)

    def test_not_warm_raises(self):
        # no financials block (account=None) => not connected/warm
        with self.assertRaises(ValueError):
            _broker(_FakeClient(_detail(account=None)))._get_account()

    def test_sync_positions_and_orders(self):
        positions = [
            SimpleNamespace(symbol="EURUSDm", volume=0.10, side="BUY", entry_price=1.10, current_price=1.11),
            SimpleNamespace(symbol="GBPUSDm", volume=0.20, side="SELL", entry_price=1.30, current_price=1.29),
        ]
        pending = [SimpleNamespace(symbol="EURUSDm", volume=0.01, side="BUY", ticket="42", limit_price=1.05, price=1.05)]
        account = _broker(_FakeClient(_detail(_fin(balance=1000.0), positions), pending))._get_account()
        self.assertEqual(len(account.portfolio), 2)
        self.assertEqual(account.portfolio[_to_asset("EURUSDm")].size, Decimal("0.1"))
        self.assertTrue(account.portfolio[_to_asset("GBPUSDm")].is_short)
        self.assertEqual(len(account.orders), 1)
        self.assertEqual(account.orders[0].id, "42")
        self.assertTrue(account.orders[0].is_buy)


if __name__ == "__main__":
    unittest.main()
