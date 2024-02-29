from roboquant import Timeframe, Config
from roboquant.feeds import TiingoLiveFeed
import logging
import unittest

from tests.common import run_priceitem_feed


class TestTiingo(unittest.TestCase):

    def setUp(self):
        logging.basicConfig(level=logging.INFO)
        config = Config()
        self.key = config.get("tiingo.key")

    def test_tiingo_crypt_live_feed(self):
        feed = TiingoLiveFeed(self.key)
        feed.subscribe("btcusdt", "ethusdt")
        run_priceitem_feed(feed, ["BTCUSDT", "ETHUSDT"], self, Timeframe.next(minutes=1))
        feed.close()

    def test_tiingo_fx_live_feed(self):
        feed = TiingoLiveFeed(self.key, "fx")
        feed.subscribe("eurusd")
        run_priceitem_feed(feed, ["EURUSD"], self, Timeframe.next(minutes=1))
        feed.close()

    def test_tiingo_iex_live_feed(self):
        feed = TiingoLiveFeed(self.key, "iex")
        feed.subscribe("IBM", "TSLA")
        run_priceitem_feed(feed, ["IBM", "TSLA"], self, Timeframe.next(minutes=1))
        feed.close()


if __name__ == "__main__":
    unittest.main()
