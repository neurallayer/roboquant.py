from datetime import date, datetime, timedelta
import pathlib
from unittest import TestCase
from roboquant import Feed, PriceItem, Candle, Quote, Trade, EventChannel, CSVFeed
from roboquant.strategies.strategy import Strategy

from roboquant.feeds.feedutil import play_background
import math


def get_feed() -> CSVFeed:
    root = pathlib.Path(__file__).parent.resolve().joinpath("data", "csv")
    return CSVFeed(str(root), time_offset="21:00:00+00:00", datetime_fmt="%Y%m%d")


def get_recent_start_date(days=10):
    start = date.today() - timedelta(days=days)
    return start.strftime("%Y-%m-%d")


def get_output(filename) -> str:
    full_path = pathlib.Path(__file__).parent.resolve().joinpath("data", "output", filename)
    with open(full_path) as f:
        content = f.read()
        return content


def run_priceaction_feed(feed: Feed, symbols: list[str], testCase: TestCase, timeframe=None):
    """Common test for all feeds that produce price-bars"""

    channel = EventChannel(timeframe)
    play_background(feed, channel)

    last = None
    while event := channel.get(10.0):

        testCase.assertIsInstance(event.time, datetime)
        testCase.assertEqual("UTC", event.time.tzname())

        if last is not None:
            # testCase.assertLessEqual(event.time - last, timedelta(minutes=1))
            testCase.assertGreaterEqual(event.time, last)

        last = event.time

        for action in event.items:
            testCase.assertIsInstance(action, PriceItem)
            testCase.assertIn(action.symbol, symbols)
            testCase.assertEqual(action.symbol.upper(), action.symbol)

            match action:
                case Candle():
                    ohlcv = action.ohlcv
                    for i in range(0, 4):
                        testCase.assertGreaterEqual(ohlcv[1], ohlcv[i])  # High >= OHLC
                        testCase.assertGreaterEqual(ohlcv[i], ohlcv[2])  # OHLC >= Low
                case Trade():
                    testCase.assertTrue(math.isfinite(action.trade_price))
                    testCase.assertTrue(math.isfinite(action.trade_volume))
                case Quote():
                    for f in action.data:
                        testCase.assertTrue(math.isfinite(f))


def run_strategy(strategy: Strategy, testCase: TestCase):
    feed = get_feed()
    channel = EventChannel()
    play_background(feed, channel)
    tot_ratings = 0
    while event := channel.get():
        ratings = strategy.give_ratings(event)
        for symbol, rating in ratings.items():
            testCase.assertEqual(type(rating), float)
            testCase.assertEqual(type(symbol), str)
            testCase.assertEqual(symbol, symbol.upper())
            testCase.assertTrue(-1.0 <= rating <= 1.0)
            testCase.assertIn(symbol, feed.symbols)
        tot_ratings += len(ratings)

    testCase.assertGreater(tot_ratings, 0)
