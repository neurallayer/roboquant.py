import math
import pathlib
from datetime import date, datetime, timedelta
from typing import Iterable
from unittest import TestCase

from roboquant import PriceItem, Bar, Quote, Trade
from roboquant.account import Account
from roboquant.asset import Asset
from roboquant.feeds import CSVFeed
from roboquant.feeds.feed import Feed
from roboquant.order import Order
from roboquant.strategies.strategy import Strategy
from roboquant.wallet import Wallet, Amount


def get_feed() -> CSVFeed:
    root = pathlib.Path(__file__).parent.resolve().joinpath("data", "yahoo")
    return CSVFeed(str(root), time_offset="21:00:00+00:00")


def get_recent_start_date(days=10):
    start = date.today() - timedelta(days=days)
    return start.strftime("%Y-%m-%d")


def run_price_item_feed(feed: Feed, assets: Iterable[Asset], test_case: TestCase, timeframe=None, min_items=1):
    """Common test for all feeds that produce price-items"""

    channel = feed.play_background(timeframe)

    last = datetime.fromisoformat("1900-01-01T00:00:00+00:00")
    n_items = 0
    while event := channel.get(30.0):
        test_case.assertIsInstance(event.time, datetime)
        test_case.assertEqual("UTC", event.time.tzname())
        test_case.assertGreater(event.time, last, f"{event} < {last}, items={event.items}")
        last = event.time

        n_items += len(event.items)

        for item in event.items:
            test_case.assertIsInstance(item, PriceItem)
            test_case.assertIsInstance(item.asset, Asset)
            test_case.assertIn(item.asset, assets)

            match item:
                case Bar():
                    ohlcv = item.ohlcv
                    v = ohlcv[4]
                    test_case.assertTrue(math.isnan(v) or v >= 0.0)
                    for i in range(0, 4):
                        test_case.assertGreaterEqual(ohlcv[1], ohlcv[i])  # High >= OHLC
                        test_case.assertGreaterEqual(ohlcv[i], ohlcv[2])  # OHLC >= Low
                case Trade():
                    test_case.assertTrue(math.isfinite(item.trade_price))
                    test_case.assertTrue(math.isfinite(item.trade_volume))
                case Quote():
                    for f in item.data:
                        test_case.assertTrue(math.isfinite(f))
                    test_case.assertGreaterEqual(item.data[0], item.data[2])  # ask >= bid

    test_case.assertGreaterEqual(n_items, min_items)


def run_strategy(strategy: Strategy, test_case: TestCase):
    feed = get_feed()
    channel = feed.play_background()
    total_orders = 0
    account = Account()
    account.cash = Wallet(Amount("USD", 1_000_000.0))
    account.buying_power = Amount("USD", 1_000_000.0)
    while event := channel.get():
        orders = strategy.create_orders(event, account)
        for order in orders:
            asset = order.asset
            test_case.assertEqual(type(order), Order)
            test_case.assertEqual(asset.symbol, asset.symbol.upper())
            test_case.assertIn(asset, feed.assets)
        total_orders += len(orders)

    test_case.assertGreater(total_orders, 0)
