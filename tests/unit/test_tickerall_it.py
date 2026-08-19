"""
Live integration tests for the TickerAll broker and feeds.

Runs when `TICKERALL_API_KEY` is set together with EITHER a MetaTrader login (`TICKERALL_SERVER` /
`TICKERALL_ACCOUNT` / `TICKERALL_PASSWORD`) or a pre-connected `TICKERALL_ACCOUNT_ID`; otherwise every
test is skipped so the normal CI build stays green without credentials. Given only the login, the suite
opens a session and derives the TickerAll `account_id` itself — so you never look up the internal id (a
broker account NUMBER is not it). The connected account must be a demo account; a free TickerAll account
can connect a broker demo and run this whole suite.

`TestTickerAllConnectIT` additionally exercises the credential `connect(...)` path directly.

    # from a MetaTrader login (recommended — no internal account_id needed):
    TICKERALL_API_KEY=...  TICKERALL_BROKER=mt5  TICKERALL_SERVER=Exness-MT5Trial \
        TICKERALL_ACCOUNT=12345678  TICKERALL_PASSWORD=...  [TICKERALL_SYMBOL=EURUSDm]  [TICKERALL_TEST_TRADE=1] \
        python -m unittest tests.unit.test_tickerall_it -v

    # or from an already-connected TickerAll account_id:
    TICKERALL_API_KEY=...  TICKERALL_ACCOUNT_ID=<your-tickerall-account-id>  [TICKERALL_SYMBOL=EURUSDm] \
        python -m unittest tests.unit.test_tickerall_it -v
"""
import os
import time
import unittest
from datetime import timedelta
from decimal import Decimal
from typing import cast

from tickerall.types import BrokerName

from roboquant.common.event import Bar, Quote
from roboquant.common.order import Order
from roboquant.common.timeframe import Timeframe
from roboquant.tickerall import TickerAllBroker, TickerAllHistoricFeed, TickerAllLiveFeed
from roboquant.tickerall.tickerall_feed import _to_asset

KEY = os.environ.get("TICKERALL_API_KEY", "")
ACCOUNT_ID = os.environ.get("TICKERALL_ACCOUNT_ID", "")
SYMBOL = os.environ.get("TICKERALL_SYMBOL", "EURUSDm")
BROKER = os.environ.get("TICKERALL_BROKER", "")
SERVER = os.environ.get("TICKERALL_SERVER", "")
ACCOUNT = os.environ.get("TICKERALL_ACCOUNT", "")
PASSWORD = os.environ.get("TICKERALL_PASSWORD", "")


@unittest.skipUnless(
    KEY and (ACCOUNT_ID or (SERVER and ACCOUNT and PASSWORD)),
    "set TICKERALL_API_KEY and either the MetaTrader login "
    "(TICKERALL_SERVER/TICKERALL_ACCOUNT/TICKERALL_PASSWORD) or a pre-connected TICKERALL_ACCOUNT_ID",
)
class TestTickerAllIT(unittest.TestCase):
    # The TickerAll account_id every test reads/trades. Resolved ONCE here so the suite runs from a
    # MetaTrader login alone — you never look up the internal id. A non-numeric TICKERALL_ACCOUNT_ID
    # (an already-connected id) is used as-is; otherwise the login opens a session and we reuse its id.
    account_id: str = ""
    _owner: "TickerAllBroker | None" = None

    @classmethod
    def setUpClass(cls) -> None:
        if ACCOUNT_ID and not ACCOUNT_ID.isdigit():
            cls.account_id = ACCOUNT_ID
        elif SERVER and ACCOUNT and PASSWORD:
            cls._owner = TickerAllBroker.connect(
                KEY, broker=cast(BrokerName, BROKER or "mt5"), server=SERVER, account=ACCOUNT, password=PASSWORD
            )
            cls.account_id = cls._owner.account_id
        else:
            cls.account_id = ACCOUNT_ID  # numeric-only id, no login → the constructor guard explains

    @classmethod
    def tearDownClass(cls) -> None:
        if cls._owner is not None:
            cls._owner.close()  # ends the session setUpClass opened

    def test_sync_account(self):
        broker = TickerAllBroker(KEY, self.account_id)
        self.addCleanup(broker.close)
        account = broker.sync()
        # a MetaTrader account has a single deposit currency; a 0.0 balance is a valid state
        self.assertLessEqual(len(account.cash), 1)
        self.assertGreaterEqual(account.buying_power.value, 0.0)
        # placing no orders is a no-op and must not raise
        broker.place_orders([])
        self.assertGreaterEqual(broker.sync().buying_power.value, 0.0)

    def test_historic_candles(self):
        feed = TickerAllHistoricFeed(KEY, self.account_id)
        self.addCleanup(feed.close)
        feed.retrieve(SYMBOL, timeframe="H1", hours=168)
        if not feed.assets():
            self.skipTest("no candles returned (market closed or symbol unavailable)")
        bars = [item for event in feed.play() for item in event.items if isinstance(item, Bar)]
        self.assertTrue(bars, "expected at least one Bar")
        self.assertTrue(all(b.asset.symbol == SYMBOL for b in bars))
        times = [event.time for event in feed.play()]
        self.assertEqual(times, sorted(times), "bars must be time-ordered")

    def test_live_ticks(self):
        feed = TickerAllLiveFeed(KEY, self.account_id)
        feed.subscribe(SYMBOL)
        quotes = []
        try:
            for event in feed.play(Timeframe.next(timedelta(seconds=25))):
                quotes.extend(i for i in event.items if isinstance(i, Quote))
                if len(quotes) >= 3:
                    break
        finally:
            feed.close()
        if not quotes:
            self.skipTest("no ticks received (market closed?)")
        self.assertEqual(quotes[0].asset.symbol, SYMBOL)
        self.assertGreater(quotes[0].ask_price, 0.0)

    @unittest.skipUnless(os.environ.get("TICKERALL_TEST_TRADE"), "set TICKERALL_TEST_TRADE=1 to run live trades")
    def test_trade_round_trip(self):
        broker = TickerAllBroker(KEY, self.account_id)
        self.addCleanup(broker.close)
        client = broker.client
        aid = broker.account_id
        # never place an order on anything other than a demo account
        self.assertIs(client.accounts.get(aid).is_demo, True, "refusing to trade a non-demo account")
        asset = _to_asset(SYMBOL)

        def position_tickets():
            return {str(p.ticket) for p in client.accounts.get(aid).positions if p.ticket}

        def pending_by_ticket():
            return {str(o.ticket): o for o in client.orders.list_pending(aid) if o.ticket}

        # --- market place -> close ---
        before = position_tickets()
        broker.place_orders([Order(asset, Decimal("0.01"))])  # no limit → market
        time.sleep(4)
        opened = position_tickets() - before
        self.assertTrue(opened, "a market order should open a position")
        for ticket in opened:
            client.positions.close(aid, int(ticket))
        time.sleep(4)
        self.assertFalse(position_tickets() & opened, "the opened position(s) must be closed")

        # --- pending place -> modify -> cancel ---
        candles = client.candles.get(aid, symbol=SYMBOL, count=6, timeframe="M1")
        if not candles:
            self.skipTest("no price available to place a resting pending order")
        last = float(candles[-1].close)
        p1, p2 = round(last * 0.90, 5), round(last * 0.85, 5)

        before_pending = set(pending_by_ticket())
        broker.place_orders([Order(asset, Decimal("0.01"), p1)])
        time.sleep(3)
        new_tickets = set(pending_by_ticket()) - before_pending
        self.assertEqual(len(new_tickets), 1, "exactly one new pending order expected")
        ticket = next(iter(new_tickets))

        # modify through the broker (id + non-zero size + new limit)
        resting = Order(asset, Decimal("0.01"), p1, id=ticket)
        broker.place_orders([resting.modify(limit=p2)])
        time.sleep(3)
        modified = pending_by_ticket().get(ticket)
        self.assertIsNotNone(modified, "the pending order should still be resting after modify")
        assert modified is not None  # narrow Optional for the type checker
        shown = modified.limit_price if modified.limit_price is not None else modified.price
        self.assertLess(abs(float(shown) - p2), p2 * 0.005, f"price should reflect modify (got {shown}, want ~{p2})")

        # cancel through the broker (id + zero size)
        broker.place_orders([resting.cancel()])
        time.sleep(3)
        self.assertNotIn(ticket, pending_by_ticket(), "the pending order should be cancelled")


@unittest.skipUnless(
    KEY and BROKER and SERVER and ACCOUNT and PASSWORD,
    "set TICKERALL_API_KEY + TICKERALL_BROKER/SERVER/ACCOUNT/PASSWORD to run the credential connect path",
)
class TestTickerAllConnectIT(unittest.TestCase):
    """The credential `connect(...)` path: MetaTrader login in, a working broker/feed out (read-only)."""

    def test_broker_connect_from_credentials(self):
        # connect opens the session for us (no separate sessions.start) and binds the account_id
        broker = TickerAllBroker.connect(
            KEY, broker=cast(BrokerName, BROKER), server=SERVER, account=ACCOUNT, password=PASSWORD
        )
        self.addCleanup(broker.close)  # close ends the session connect() opened
        self.assertTrue(broker.account_id, "connect must bind a TickerAll account_id")
        account = broker.sync()
        # a 0.0 balance is a valid state; only require it be readable and non-negative
        self.assertGreaterEqual(account.buying_power.value, 0.0)

        # feeds reuse the broker's session cheaply via broker.account_id (no second session opened)
        feed = TickerAllHistoricFeed(KEY, broker.account_id)
        self.addCleanup(feed.close)
        feed.retrieve(SYMBOL, timeframe="H1", hours=24)
        if not feed.assets():
            self.skipTest("no candles returned (market closed or symbol unavailable)")
        bars = [item for event in feed.play() for item in event.items if isinstance(item, Bar)]
        self.assertTrue(bars, "expected at least one Bar from the reused session")

    def test_live_feed_connect_from_credentials(self):
        # a data-only setup: no broker, connect the feed straight from credentials
        feed = TickerAllLiveFeed.connect(
            KEY, broker=cast(BrokerName, BROKER), server=SERVER, account=ACCOUNT, password=PASSWORD
        )
        self.assertTrue(feed.account_id, "connect must bind a TickerAll account_id")
        feed.subscribe(SYMBOL)
        quotes = []
        try:
            for event in feed.play(Timeframe.next(timedelta(seconds=25))):
                quotes.extend(i for i in event.items if isinstance(i, Quote))
                if len(quotes) >= 3:
                    break
        finally:
            feed.close()  # ends the session connect() opened
        if not quotes:
            self.skipTest("no ticks received (market closed?)")
        self.assertEqual(quotes[0].asset.symbol, SYMBOL)


if __name__ == "__main__":
    unittest.main()
