from roboquant import Timeframe, TiingoLiveFeed, Config
import logging
import unittest

from tests.common import run_priceaction_feed


class TestTiingo(unittest.TestCase):

    def setUp(self):
        logging.basicConfig(level=logging.INFO)
        config = Config()
        self.key = config.get("tiingo.key")

    def test_tiingo_cryptolivefeed(self):
        feed = TiingoLiveFeed(self.key)
        feed.subscribe("btcusdt", "ethusdt")
        run_priceaction_feed(feed, ["BTCUSDT", "ETHUSDT"], self, Timeframe.next(minutes=1))
        feed.close()

    def test_tiingo_fxlivefeed(self):
        feed = TiingoLiveFeed(self.key, "fx")
        feed.subscribe("eurusd")
        run_priceaction_feed(feed, ["EURUSD"], self, Timeframe.next(minutes=1))
        feed.close()

    def test_tiingo_iexlivefeed(self):
        feed = TiingoLiveFeed(self.key, "iex")
        feed.subscribe("IBM", "TSLA")
        run_priceaction_feed(feed, ["IBM", "TSLA"], self, Timeframe.next(minutes=1))
        feed.close()


if __name__ == "__main__":
    unittest.main()
