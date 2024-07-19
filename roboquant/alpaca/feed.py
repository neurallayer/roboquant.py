import threading
from array import array
from datetime import timedelta
from typing import Literal

import numpy as np
from alpaca.data import DataFeed
from alpaca.data.enums import Adjustment
from alpaca.data.historical.crypto import CryptoHistoricalDataClient
from alpaca.data.historical.stock import StockHistoricalDataClient
from alpaca.data.live.crypto import CryptoDataStream
from alpaca.data.live.option import OptionDataStream
from alpaca.data.live.stock import StockDataStream
from alpaca.data.models.bars import BarSet
from alpaca.data.models.quotes import QuoteSet
from alpaca.data.models.trades import TradeSet
from alpaca.data.requests import (
    CryptoBarsRequest,
    CryptoTradesRequest,
    StockBarsRequest,
    StockQuotesRequest,
    StockTradesRequest,
)
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.trading.enums import AssetClass

from roboquant.asset import Asset, Crypto, Option, Stock
from roboquant.config import Config
from roboquant.event import Bar, Event, Quote, Trade
from roboquant.feeds.historic import HistoricFeed
from roboquant.feeds.live import LiveFeed


def _get_asset(symbol: str, asset_class: AssetClass) -> Asset:
    match asset_class:
        case AssetClass.US_EQUITY:
            return Stock(symbol, "USD")
        case AssetClass.CRYPTO:
            return Crypto.from_symbol(symbol)
        case AssetClass.US_OPTION:
            return Option(symbol, "USD")


class AlpacaLiveFeed(LiveFeed):
    """Subscribe to live market data for stocks, crypto currencies or options"""

    __one_minute = str(timedelta(minutes=1))

    def __init__(self, market: Literal["iex", "sip", "crypto", "option"] = "iex", api_key=None, secret_key=None) -> None:
        super().__init__()
        config = Config()
        api_key = api_key or config.get("alpaca.public.key")
        secret_key = secret_key or config.get("alpaca.secret.key")
        self.market = market

        match market:
            case "sip":
                self.stream = StockDataStream(api_key, secret_key, feed=DataFeed.SIP)
                self.asset_class = AssetClass.US_EQUITY
            case "iex":
                self.stream = StockDataStream(api_key, secret_key, feed=DataFeed.IEX)
                self.asset_class = AssetClass.US_EQUITY
            case "crypto":
                self.stream = CryptoDataStream(api_key, secret_key)
                self.asset_class = AssetClass.CRYPTO
            case "option":
                self.stream = OptionDataStream(api_key, secret_key)
                self.asset_class = AssetClass.US_OPTION
            case _:
                raise ValueError(f"unsupported value market={market}")

        thread = threading.Thread(None, self.stream.run, daemon=True)
        thread.start()

    async def close(self):
        await self.stream.close()

    def __put_item(self, time, item):
        event = Event(time, [item])
        self.put(event)

    async def __handle_trades(self, data):
        asset = _get_asset(data.symbol, self.asset_class)
        item = Trade(asset, data.price, data.size)
        self.__put_item(data.timestamp, item)

    async def __handle_bars(self, data):
        asset = _get_asset(data.symbol, self.asset_class)
        item = Bar(asset, array("f", [data.open, data.high, data.low, data.close, data.volume]), self.__one_minute)
        self.__put_item(data.timestamp, item)

    async def __handle_quotes(self, data):
        asset = _get_asset(data.symbol, self.asset_class)
        item = Quote(asset, array("f", [data.ask_price, data.ask_size, data.bid_price, data.bid_size]))
        self.__put_item(data.timestamp, item)

    def subscribe_trades(self, *symbols: str):
        self.stream.subscribe_trades(self.__handle_trades, *symbols)

    def subscribe_quotes(self, *symbols: str):
        self.stream.subscribe_quotes(self.__handle_quotes, *symbols)

    def subscribe_bars(self, *symbols: str):
        self.stream.subscribe_bars(self.__handle_bars, *symbols)


class _AlpacaHistoricFeed(HistoricFeed):

    def _process_bars(self, bar_set, freq: str, asset_class: AssetClass):
        for symbol, data in bar_set.items():
            asset = _get_asset(symbol, asset_class)
            for d in data:
                time = d.timestamp
                ohlcv = array("f", [d.open, d.high, d.low, d.close, d.volume])
                item = Bar(asset, ohlcv, freq)
                super()._add_item(time, item)

    def _process_trades(self, quote_set, asset_class):
        for symbol, data in quote_set.items():
            asset = _get_asset(symbol, asset_class)
            for d in data:
                time = d.timestamp
                item = Trade(asset, d.price, d.size)
                super()._add_item(time, item)

    def _process_quotes(self, quote_set, asset_class):
        for symbol, data in quote_set.items():
            asset = _get_asset(symbol, asset_class)
            for d in data:
                time = d.timestamp
                arr = array("f", [d.ask_price, d.ask_size, d.bid_price, d.bid_size])

                if np.all(arr):
                    # on rare occasions values are missing and have 0.0 as a value
                    item = Quote(asset, arr)
                    super()._add_item(time, item)


class AlpacaHistoricStockFeed(_AlpacaHistoricFeed):
    """Get historic stock prices from Alpaca.
    Support for bars, trades and quotes.
    """

    def __init__(self, api_key=None, secret_key=None, data_api_url=None, feed: DataFeed | None = None):
        super().__init__()
        config = Config()
        api_key = api_key or config.get("alpaca.public.key")
        secret_key = secret_key or config.get("alpaca.secret.key")
        self.client = StockHistoricalDataClient(api_key, secret_key, url_override=data_api_url)
        self.feed = feed

    def retrieve_bars(self, *symbols, start=None, end=None, resolution: TimeFrame | None = None, adjustment=Adjustment.ALL):
        resolution = resolution or TimeFrame(amount=1, unit=TimeFrameUnit.Day)
        req = StockBarsRequest(
            symbol_or_symbols=list(symbols), timeframe=resolution, start=start, end=end, adjustment=adjustment, feed=self.feed
        )
        res = self.client.get_stock_bars(req)
        assert isinstance(res, BarSet)
        freq = str(resolution)
        self._process_bars(res.data, freq, AssetClass.US_EQUITY)

    def retrieve_trades(self, *symbols, start=None, end=None):
        req = StockTradesRequest(symbol_or_symbols=list(symbols), start=start, end=end, feed=self.feed)
        res = self.client.get_stock_trades(req)
        assert isinstance(res, TradeSet)
        self._process_trades(res.data, AssetClass.US_EQUITY)

    def retrieve_quotes(self, *symbols: str, start=None, end=None):
        req = StockQuotesRequest(symbol_or_symbols=list(symbols), start=start, end=end, feed=self.feed)
        res = self.client.get_stock_quotes(req)
        assert isinstance(res, QuoteSet)
        self._process_quotes(res.data, AssetClass.US_EQUITY)


class AlpacaHistoricCryptoFeed(_AlpacaHistoricFeed):
    """Get historic crypto-currency prices from Alpaca.
    Support for bars and trades.
    """

    def __init__(self, api_key=None, secret_key=None, data_api_url=None):
        super().__init__()
        config = Config()
        api_key = api_key or config.get("alpaca.public.key")
        secret_key = secret_key or config.get("alpaca.secret.key")
        self.client = CryptoHistoricalDataClient(api_key, secret_key, url_override=data_api_url)

    def retrieve_bars(self, *symbols, start=None, end=None, resolution: TimeFrame | None = None):
        resolution = resolution or TimeFrame(amount=1, unit=TimeFrameUnit.Day)
        req = CryptoBarsRequest(symbol_or_symbols=list(symbols), timeframe=resolution, start=start, end=end)
        res = self.client.get_crypto_bars(req)
        assert isinstance(res, BarSet)
        freq = str(resolution)
        self._process_bars(res.data, freq, AssetClass.CRYPTO)

    def retrieve_trades(self, *symbols, start=None, end=None):
        req = CryptoTradesRequest(symbol_or_symbols=list(symbols), start=start, end=end)
        res = self.client.get_crypto_trades(req)
        assert isinstance(res, TradeSet)
        self._process_trades(res.data, AssetClass.CRYPTO)
