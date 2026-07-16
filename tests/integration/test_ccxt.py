import unittest
import os
import ccxt
from datetime import datetime

from roboquant.asset import Crypto
from roboquant.crypto.cryptofeed import CryptoFeed
from roboquant.crypto.cryptobroker import CryptoBroker
from tests.common import run_price_item_feed

from dotenv import load_dotenv

load_dotenv()


class TestCCXT(unittest.TestCase):

    def test_binance_feed(self):
        symbols = {"BTC/USDT", "ETH/USDT"}
        binance = ccxt.binance()
        feed = CryptoFeed(binance, *symbols, start_date="2024-01-01T00:00:00", end_date="2025-01-01T00:00:00")
        self.assertEqual(2, len(feed.assets()))
        self.assertEqual(symbols, {a.symbol for a in feed.assets()})

        assets = {Crypto.from_symbol(symbol) for symbol in symbols}
        self.assertEqual(assets, set(feed.assets()))

        tf = feed.timeframe()
        assert tf
        self.assertTrue(tf.start == datetime.fromisoformat("2024-01-01 00:00:00+00:00"))
        self.assertTrue(tf.end == datetime.fromisoformat("2024-12-31 00:00:00+00:00"))
        self.assertEqual(366, len(feed.timeline()))
        run_price_item_feed(feed, assets, self)

    def test_alpaca_broker(self):
        key = os.getenv("ALPACA_API_KEY")
        secret = os.getenv("ALPACA_SECRET")
        assert(key is not None and secret is not None), "ALPACA_API_KEY and ALPACA_SECRET must be set"
        alpaca = ccxt.alpaca({
            "apiKey": key,
            "secret": secret
        })
        alpaca.set_sandbox_mode(True)
        broker = CryptoBroker(alpaca)
        account = broker.sync()
        print(account)

if __name__ == "__main__":
    unittest.main()
