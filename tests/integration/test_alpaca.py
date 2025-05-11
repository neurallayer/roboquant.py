import time
import unittest
import os
from alpaca.data.timeframe import TimeFrame

from roboquant.alpaca import AlpacaHistoricCryptoFeed, AlpacaHistoricStockFeed, AlpacaBroker
from roboquant.asset import Crypto, Stock
from roboquant.order import Order
from tests.common import run_price_item_feed
from dotenv import load_dotenv

load_dotenv()


def _get_credentials():
    return os.environ["ALPACA_API_KEY"], os.environ["ALPACA_SECRET"]

class TestAlpaca(unittest.TestCase):

    stocks = ["AAPL", "TSLA"]
    assets = [Stock(symbol) for symbol in stocks]
    cryptos = [Crypto.from_symbol("BTC/USDT")]

    def test_alpaca_stock_feed_bars(self):
        feed = AlpacaHistoricStockFeed(*_get_credentials())
        feed.retrieve_bars(*self.stocks, start="2024-03-01", end="2024-03-02")
        run_price_item_feed(feed, self.assets, self)

    def test_alpaca_stock_feed_trades(self):
        feed = AlpacaHistoricStockFeed(*_get_credentials())
        feed.retrieve_trades(*self.stocks, start="2024-03-01T18:00:00", end="2024-03-01T18:01:00")
        run_price_item_feed(feed, self.assets, self)

    def test_alpaca_stock_feed_quotes(self):
        feed = AlpacaHistoricStockFeed(*_get_credentials())
        feed.retrieve_quotes(*self.stocks, start="2024-03-01T18:00:00", end="2024-03-01T18:01:00")
        run_price_item_feed(feed, self.assets, self)

    def test_alpaca_crypto_feed_bars(self):
        feed = AlpacaHistoricCryptoFeed(*_get_credentials())
        feed.retrieve_bars("BTC/USDT", start="2024-03-01", end="2024-03-02", resolution=TimeFrame.Hour)  # type: ignore
        run_price_item_feed(feed, self.cryptos, self)

    def test_alpaca_crypto_feed_trades(self):
        feed = AlpacaHistoricCryptoFeed(*_get_credentials())
        feed.retrieve_trades("BTC/USDT", start="2024-04-01", end="2024-04-10")
        run_price_item_feed(feed, self.cryptos, self)

    def test_alpaca_broker(self):
        broker = AlpacaBroker(*_get_credentials())
        account = broker.sync()
        print(account)
        self.assertTrue(account.buying_power.value > 0)
        self.assertTrue(account.equity_value() > 0)
        order = Order(self.assets[0], 1, 100.0)
        print(order)
        broker.place_orders([order])
        time.sleep(5)
        account = broker.sync()
        print(account)
        cancel_orders = [order.cancel() for order in account.orders]
        print(cancel_orders)
        broker.place_orders(cancel_orders)
        time.sleep(10)
        print(account)


if __name__ == "__main__":
    unittest.main()
