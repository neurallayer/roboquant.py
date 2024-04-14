import logging
import unittest

from roboquant import Config, Timeframe
from roboquant.feeds import TiingoHistoricFeed, TiingoLiveFeed
from tests.common import get_recent_start_date, run_price_item_feed


class TestTiingo(unittest.TestCase):

    def setUp(self) -> None:
        config = Config()
        self.key = config.get("tiingo.key")
        logging.basicConfig(level=logging.INFO)

    def test_tiingo_historic_eod(self):
        feed = TiingoHistoricFeed(self.key)
        feed.retrieve_eod_stocks("TSLA", "AMZN", "INVALID_TICKER_NAME", start_date="2020-01-01")
        run_price_item_feed(feed, ["TSLA", "AMZN"], self)

    def test_tiingo_historic_intra(self):
        feed = TiingoHistoricFeed(self.key)
        feed.retrieve_intraday_iex("TSLA", start_date=get_recent_start_date())
        run_price_item_feed(feed, ["TSLA"], self)

    def test_tiingo_historic_crypto(self):
        feed = TiingoHistoricFeed(self.key)
        feed.retrieve_intraday_crypto("BTCUSD", start_date=get_recent_start_date())
        run_price_item_feed(feed, ["BTCUSD"], self)

    def test_tiingo_historic_fx(self):
        feed = TiingoHistoricFeed(self.key)
        feed.retrieve_intraday_fx("EURUSD", "AUDUSD", start_date=get_recent_start_date())
        run_price_item_feed(feed, ["EURUSD", "AUDUSD"], self)

    def test_tiingo_crypt_live_feed(self):
        feed = TiingoLiveFeed(self.key)
        feed.subscribe("btcusdt", "ethusdt")
        run_price_item_feed(feed, ["BTCUSDT", "ETHUSDT"], self, Timeframe.next(minutes=1))
        feed.close()

    def test_tiingo_fx_live_feed(self):
        feed = TiingoLiveFeed(self.key, "fx")
        feed.subscribe("eurusd")
        run_price_item_feed(feed, ["EURUSD"], self, Timeframe.next(minutes=1))
        feed.close()

    def test_tiingo_iex_live_feed(self):
        feed = TiingoLiveFeed(self.key, "iex")
        feed.subscribe("IBM", "TSLA")
        run_price_item_feed(feed, ["IBM", "TSLA"], self, Timeframe.next(minutes=1))
        feed.close()


if __name__ == "__main__":
    unittest.main()
