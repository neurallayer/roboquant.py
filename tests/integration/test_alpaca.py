import unittest
from alpaca.data.timeframe import TimeFrame

from roboquant.alpaca import AlpacaHistoricCryptoFeed, AlpacaHistoricStockFeed, AlpacaBroker
from roboquant.asset import Crypto, Stock
from roboquant.order import Order
from tests.common import run_price_item_feed


class TestAlpaca(unittest.TestCase):

    stocks = ["AAPL", "TSLA"]
    assets = [Stock(symbol) for symbol in stocks]
    cryptos = [Crypto.from_symbol("BTC/USDT")]

    def test_alpaca_stock_feed_bars(self):
        feed = AlpacaHistoricStockFeed()
        feed.retrieve_bars(*self.stocks, start="2024-03-01", end="2024-03-02")
        run_price_item_feed(feed, self.assets, self)

    def test_alpaca_stock_feed_trades(self):
        feed = AlpacaHistoricStockFeed()
        feed.retrieve_trades(*self.stocks, start="2024-03-01T18:00:00", end="2024-03-01T18:01:00")
        run_price_item_feed(feed, self.assets, self)

    def test_alpaca_stock_feed_quotes(self):
        feed = AlpacaHistoricStockFeed()
        feed.retrieve_quotes(*self.stocks, start="2024-03-01T18:00:00", end="2024-03-01T18:01:00")
        run_price_item_feed(feed, self.assets, self)

    def test_alpaca_crypto_feed_bars(self):
        feed = AlpacaHistoricCryptoFeed()
        feed.retrieve_bars("BTC/USDT", start="2024-03-01", end="2024-03-02", resolution=TimeFrame.Hour)  # type: ignore
        run_price_item_feed(feed, self.cryptos, self)

    def test_alpaca_crypto_feed_trades(self):
        feed = AlpacaHistoricCryptoFeed()
        feed.retrieve_trades("BTC/USDT", start="2024-04-01", end="2024-04-10")
        run_price_item_feed(feed, self.cryptos, self)

    def test_alpaca_broker(self):
        broker = AlpacaBroker()
        account = broker.sync()
        self.assertTrue(account.buying_power.value > 0)
        self.assertTrue(account.equity_value() > 0)
        order = Order(self.assets[0], 1, 100.0)
        broker.place_orders([order])
        account = broker.sync()
        print(account)


if __name__ == "__main__":
    unittest.main()
