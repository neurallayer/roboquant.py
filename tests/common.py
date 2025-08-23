import math
import pathlib
from datetime import date, datetime, timedelta
from typing import Iterable
from unittest import TestCase

from roboquant import PriceItem, Bar, Quote, TradePrice, Timeframe
from roboquant.asset import Asset
from roboquant.feeds import CSVFeed
from roboquant.feeds.feed import Feed
from roboquant.signal import Signal
from roboquant.strategies.strategy import Strategy


def get_feed() -> CSVFeed:
    """Return a CSV feed based on some stock data in Yahoo format"""
    root = pathlib.Path(__file__).parent.resolve().joinpath("data", "yahoo")
    return CSVFeed.yahoo(str(root))

def get_recent_start_date(days: int = 10):
    """Get a recent (in the past) date"""
    start = date.today() - timedelta(days=days)
    return start.strftime("%Y-%m-%d")


def run_price_item_feed(
    feed: Feed, assets: Iterable[Asset], test_case: TestCase, timeframe: Timeframe | None = None, min_items: int = 1
):
    """Common test for all feeds that produce price-items. It validates the data and the order of the items"""

    last = datetime.fromisoformat("1900-01-01T00:00:00+00:00")
    n_items = 0
    for event in feed.play():
        test_case.assertIsInstance(event.time, datetime)
        test_case.assertEqual("UTC", event.time.tzname())
        test_case.assertGreaterEqual(event.time, last, f"{event} < {last}, items={event.items}")
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
                        test_case.assertGreaterEqual(ohlcv[i], ohlcv[2])
                        test_case.assertGreaterEqual(ohlcv[i], 0.0)
                case TradePrice():
                    test_case.assertTrue(math.isfinite(item.trade_price))
                    test_case.assertTrue(math.isfinite(item.trade_volume))
                case Quote():
                    for f in item.data:
                        test_case.assertTrue(math.isfinite(f))
                    test_case.assertGreaterEqual(item.data[0], item.data[2])  # ask >= bid
                case _:
                    pass

    test_case.assertGreaterEqual(n_items, min_items)


def run_strategy(strategy: Strategy, test_case: TestCase):
    """Run and test a strategy on the default feed"""
    feed = get_feed()
    all_assets = feed.assets()
    total_signals = 0
    for event in feed.play():
        signals = strategy.create_signals(event)
        for signal in signals:
            asset = signal.asset
            test_case.assertEqual(type(signal), Signal)
            test_case.assertEqual(asset.symbol, asset.symbol.upper())
            test_case.assertIn(asset, all_assets)
            test_case.assertGreaterEqual(signal.rating, -1)
            test_case.assertLessEqual(signal.rating, 1)
        total_signals += len(signals)

    test_case.assertGreater(total_signals, 0)
