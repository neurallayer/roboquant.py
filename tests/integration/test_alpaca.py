import unittest
from alpaca.data.timeframe import TimeFrame

from roboquant.alpaca import AlpacaHistoricCryptoFeed, AlpacaHistoricStockFeed, AlpacaBroker
from roboquant.order import Order
from tests.common import run_price_item_feed


class TestAlpaca(unittest.TestCase):

    stocks = ["AAPL", "TSLA"]

    def test_alpaca_feed(self):
        feed = AlpacaHistoricStockFeed()
        feed.retrieve_bars(*self.stocks, start="2024-03-01", end="2024-03-02")
        run_price_item_feed(feed, self.stocks, self)

        feed = AlpacaHistoricStockFeed()
        feed.retrieve_trades(*self.stocks, start="2024-03-01T18:00:00", end="2024-03-01T18:01:00")
        run_price_item_feed(feed, self.stocks, self)

        feed = AlpacaHistoricStockFeed()
        feed.retrieve_quotes(*self.stocks, start="2024-03-01T18:00:00", end="2024-03-01T18:01:00")
        run_price_item_feed(feed, self.stocks, self)

        feed = AlpacaHistoricCryptoFeed()
        feed.retrieve_bars("BTC/USDT", start="2024-03-01", end="2024-03-02", resolution=TimeFrame.Hour)  # type: ignore
        run_price_item_feed(feed, ["BTC/USDT"], self)

    def test_alpaca_broker(self):
        broker = AlpacaBroker()
        account = broker.sync()
        self.assertTrue(account.buying_power > 0)
        self.assertTrue(account.equity() > 0)
        order = Order("TSLA", 1)
        broker.place_orders([order])
        account = broker.sync()


if __name__ == "__main__":
    unittest.main()
