import logging
import unittest

from roboquant import Config
from roboquant.feeds import TiingoHistoricFeed
from tests.common import get_recent_start_date, run_priceitem_feed


class TestTiingoHistoricFeed(unittest.TestCase):

    def setUp(self) -> None:
        config = Config()
        self.key = config.get("tiingo.key")
        logging.basicConfig(level=logging.INFO)

    def test_tiingo_historic_eod(self):
        feed = TiingoHistoricFeed(self.key)
        feed.retrieve_eod_stocks("TSLA", "AMZN", "INVALID_TICKER_NAME", start_date="2020-01-01")
        run_priceitem_feed(feed, ["TSLA", "AMZN"], self)

    def test_tiingo_historic_intra(self):
        feed = TiingoHistoricFeed(self.key)
        feed.retrieve_intraday_iex("TSLA", start_date=get_recent_start_date())
        run_priceitem_feed(feed, ["TSLA"], self)

    def test_tiingo_historic_crypto(self):
        feed = TiingoHistoricFeed(self.key)
        feed.retrieve_intraday_crypto("BTCUSD", start_date=get_recent_start_date())
        run_priceitem_feed(feed, ["BTCUSD"], self)

    def test_tiingo_historic_fx(self):
        feed = TiingoHistoricFeed(self.key)
        feed.retrieve_intraday_fx("EURUSD", "AUDUSD", start_date=get_recent_start_date())
        run_priceitem_feed(feed, ["EURUSD", "AUDUSD"], self)


if __name__ == "__main__":
    unittest.main()
