"""
Live integration tests for the TickerAll broker and feeds.

These run only when `TICKERALL_API_KEY` and `TICKERALL_ACCOUNT_ID` are set in the environment; otherwise
every test is skipped so the normal CI build stays green without any credentials. The connected account must
be a demo account. A free TickerAll account can connect a broker demo and run this whole suite.

    TICKERALL_API_KEY=...  TICKERALL_ACCOUNT_ID=...  [TICKERALL_SYMBOL=EURUSDm]  [TICKERALL_TEST_TRADE=1] \
        python -m unittest tests.unit.test_tickerall_it -v
"""

import os
import time
from typing import Any
import unittest
from decimal import Decimal
from dotenv import load_dotenv

import roboquant as rq
from roboquant.common.asset import Forex
from roboquant.common.event import Bar, Quote
from roboquant.common.monetary import USD, ECBConversion
from roboquant.common.order import Order
from roboquant.common.timeframe import Timeframe
from roboquant.feeds.tickerall import TickerAllHistoricFeed, TickerAllLiveFeed, _to_asset
from roboquant.brokers.tickerall import TickerAllBroker
from roboquant.journals.basicjournal import BasicJournal
from roboquant.strategies.ema_crossover import EMACrossover
from roboquant.traders.flextrader import FlexTrader

load_dotenv()
KEY = os.environ.get("TICKERALL_API_KEY", "")
ACCOUNT_ID = os.environ.get("TICKERALL_ACCOUNT_ID", "")
SYMBOL = "BTCUSD"

MT5_SERVER = os.environ.get("MT5_SERVER")
MT5_ACCOUNT = os.environ.get("MT5_ACCOUNT")
MT5_PASSWORD = os.environ.get("MT5_PASSWORD")

CONFIG: dict[str, Any] = {"broker": "mt5", "server": MT5_SERVER, "account": MT5_ACCOUNT, "password": MT5_PASSWORD}


@unittest.skipUnless(
    KEY and ACCOUNT_ID and MT5_SERVER and MT5_ACCOUNT and MT5_PASSWORD, "set TICKERALL_API_KEY and TICKERALL_ACCOUNT_ID to run"
)
class TestTickerAllIT(unittest.TestCase):
    def test_live_ticks(self):
        feed = TickerAllLiveFeed.connect(KEY, **CONFIG)
        feed.subscribe(SYMBOL)
        quotes: list[Quote] = []
        try:
            for event in feed.play(Timeframe.next("60 seconds")):
                quotes.extend(i for i in event.items if isinstance(i, Quote))
                print(event.items, flush=True)
        finally:
            feed.close()
        if not quotes:
            self.skipTest("no ticks received (market closed?)")
        self.assertEqual(quotes[0].asset.symbol, SYMBOL)
        self.assertGreater(quotes[0].ask_price, 0.0)

    def test_sync_account(self):
        broker = TickerAllBroker.connect(KEY, **CONFIG)
        self.addCleanup(broker.close)
        account = broker.sync()
        print(account)

        for asset, position in account.portfolio.items():
            print(asset, position)

        if account.orders:
            cancellations = [order.cancel() for order in account.orders]
            broker.place_orders(cancellations)
            time.sleep(10)
            account = broker.sync()
            print(account)

        asset = Forex("BTCUSD", USD)
        price = account.portfolio[asset].mkt_price
        print("market price =", price, flush=True)
        order = Order(asset, Decimal("0.001"), limit=round(price*1.0001, 2))
        broker.place_orders([order])
        for _ in range(5):
            account = broker.sync()
            print(account, "\n", flush=True)
            time.sleep(10)

    def test_full_paper_trade(self):
        broker = None
        feed = None
        try:
            broker = TickerAllBroker.connect(KEY, **CONFIG)
            feed = TickerAllLiveFeed(KEY, broker.account_id)
            feed.subscribe(SYMBOL)
            strategy = EMACrossover(2, 5)
            trader = FlexTrader(
                size_fractions=4,
                shorting=True,
                max_order_pct=0.01,
                min_order_pct=0.001,
                max_position_pct=0.2,
                limit_rounding=6
            )
            tf = rq.Timeframe.next("60 minutes")
            journal = BasicJournal()
            ECBConversion().register()
            account = rq.run(feed, strategy, trader, broker=broker, journal=journal, timeframe=tf)
            print(account)
        finally:
            if broker:
                broker.close()
            if feed:
                feed.close()

    def test_historic_candles(self):
        feed = TickerAllHistoricFeed.connect(KEY, **CONFIG)
        self.addCleanup(feed.close)
        feed.retrieve(SYMBOL, timeframe="H1", hours=168)
        print(feed)
        if not feed.assets():
            self.skipTest("no candles returned (market closed or symbol unavailable)")
        bars = [item for event in feed.play() for item in event.items if isinstance(item, Bar)]
        self.assertTrue(bars, "expected at least one Bar")
        self.assertTrue(all(b.asset.symbol == SYMBOL for b in bars))
        times = [event.time for event in feed.play()]
        self.assertEqual(times, sorted(times), "bars must be time-ordered")

    @unittest.skipUnless(os.environ.get("TICKERALL_TEST_TRADE"), "set TICKERALL_TEST_TRADE=1 to run live trades")
    def test_trade_round_trip(self):
        broker = TickerAllBroker(KEY, ACCOUNT_ID)
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
        broker.place_orders([Order(asset, Decimal("0.01"), float("nan"))])
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
        assert modified
        self.assertIsNotNone(modified, "the pending order should still be resting after modify")
        shown = modified.limit_price if modified.limit_price is not None else modified.price
        self.assertLess(abs(float(shown) - p2), p2 * 0.005, f"price should reflect modify (got {shown}, want ~{p2})")

        # cancel through the broker (id + zero size)
        broker.place_orders([resting.cancel()])
        time.sleep(3)
        self.assertNotIn(ticket, pending_by_ticket(), "the pending order should be cancelled")


if __name__ == "__main__":
    unittest.main()
