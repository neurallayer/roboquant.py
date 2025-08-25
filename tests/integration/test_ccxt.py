import unittest
import ccxt
from datetime import datetime

from roboquant.asset import Crypto
from roboquant.feeds.cryptofeed import CryptoFeed
from tests.common import run_price_item_feed


class TestCCXT(unittest.TestCase):

    def test_binance_feed(self):
        symbols = {"BTC/USDT", "ETH/USDT"}
        binance = ccxt.binance()
        feed = CryptoFeed(binance, *symbols, start_date="2024-01-01T00:00:00", end_date="2025-01-01T00:00:00")
        print(feed)
        self.assertEqual(2, len(feed.assets()))

        assets = {Crypto.from_symbol(symbol) for symbol in symbols}
        self.assertEqual(assets, set(feed.assets()))

        tf = feed.timeframe()
        assert tf
        self.assertTrue(tf.start == datetime.fromisoformat("2024-01-01 00:00:00+00:00"))
        self.assertTrue(tf.end == datetime.fromisoformat("2024-12-31 00:00:00+00:00"))
        self.assertEqual(366, len(feed.timeline()))
        run_price_item_feed(feed, assets, self)


if __name__ == "__main__":
    unittest.main()
