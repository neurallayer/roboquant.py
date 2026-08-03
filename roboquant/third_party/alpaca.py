from decimal import Decimal
import logging
import threading
from array import array
from datetime import datetime, timedelta
from typing import Any, Literal, override

from alpaca.trading.client import TradingClient
from alpaca.trading.models import Order as AOrder, Position as APosition, TradeAccount
from alpaca.trading.requests import GetOrdersRequest, LimitOrderRequest, ReplaceOrderRequest
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
from alpaca.trading.enums import AssetClass, OrderSide, PositionSide, QueryOrderStatus, TimeInForce

from roboquant.brokers.livebroker import LiveBroker
from roboquant.common.account import Account
from roboquant.common.asset import Asset, Crypto, Option, Stock
from roboquant.common.event import Bar, Event, PriceItem, Quote, TradePrice
from roboquant.common.monetary import USD, Amount, Wallet
from roboquant.common.order import Order
from roboquant.common.portfolio import Portfolio, Position
from roboquant.feeds.in_memory_feed import InMemoryFeed
from roboquant.feeds.livefeed import LiveFeed


logger = logging.getLogger(__name__)


def _get_asset(symbol: str, asset_class: AssetClass) -> Asset:
    """Convert an Alpaca asset to a roboquant asset based on
    its symbol name and asset class.
    """

    match asset_class:
        case AssetClass.US_EQUITY:
            return Stock(symbol)
        case AssetClass.CRYPTO:
            return Crypto.from_symbol(symbol)
        case AssetClass.US_OPTION:
            return Option(symbol)
        case AssetClass.CRYPTO_PERP:
            return Crypto.from_symbol(symbol)


class AlpacaLiveFeed(LiveFeed):
    """Subscribe to live market data for stocks, cryptocurrencies or options.

    Args:
        api_key (str): The API key for Alpaca.
        secret_key (str): The secret key for Alpaca.
        market (Literal["iex", "sip", "crypto", "option"], optional): The market to subscribe to. Defaults to "iex".
    """

    __one_minute = str(timedelta(minutes=1))

    def __init__(self, api_key: str, secret_key: str, market: Literal["iex", "sip", "crypto", "option"] = "iex") -> None:
        super().__init__()

        assert market in ["iex", "sip", "crypto", "option"], "invalid market"

        self.stream: StockDataStream | CryptoDataStream | OptionDataStream

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

        thread = threading.Thread(None, self.stream.run, daemon=True)
        thread.start()

    async def close(self):
        """Close the live feed connection stream."""
        await self.stream.close()

    def __put_item(self, time: datetime, item: PriceItem):
        event = Event(time, [item])
        self._put(event)

    async def __handle_trades(self, data):
        asset = _get_asset(data.symbol, self.asset_class)
        item = TradePrice(asset, data.price, data.size)
        self.__put_item(data.timestamp, item)

    async def __handle_bars(self, data):
        asset = _get_asset(data.symbol, self.asset_class)
        item = Bar(asset, array("f", [data.open, data.high, data.low, data.close, data.volume]), self.__one_minute)
        self.__put_item(data.timestamp, item)

    async def __handle_quotes(self, data):
        asset = _get_asset(data.symbol, self.asset_class)
        item = Quote(asset, array("f", [data.ask_price, data.ask_size, data.bid_price, data.bid_size]))
        self.__put_item(data.timestamp, item)

    @override
    def assets(self) -> list[Asset]:
        return []

    def subscribe_trades(self, *symbols: str):
        """Subscribe to trade data for the given symbols.

        Args:
            *symbols (str): The symbols to subscribe to.
        """
        self.stream.subscribe_trades(self.__handle_trades, *symbols)

    def subscribe_quotes(self, *symbols: str):
        """Subscribe to quote data for the given symbols.

        Args:
            *symbols (str): The symbols to subscribe to.
        """
        self.stream.subscribe_quotes(self.__handle_quotes, *symbols)

    def subscribe_bars(self, *symbols: str):
        """Subscribe to bar data for the given symbols.

        Args:
            *symbols (str): The symbols to subscribe to.
        """
        if not isinstance(self.stream, OptionDataStream):
            self.stream.subscribe_bars(self.__handle_bars, *symbols)
        else:
            logger.warning("cannot subscribe to bars for options")


class _AlpacaHistoricFeed(InMemoryFeed):
    """Base class for Alpaca historic feeds.
    This class is not intended to be used directly."""

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
                item = TradePrice(asset, d.price, d.size)
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

    Args:
        api_key (str): The API key for Alpaca.
        secret_key (str): The secret key for Alpaca.
        feed (DataFeed, optional): The data feed to use. Defaults to DataFeed.IEX.
        **kwargs: Additional keyword arguments.
    """

    def __init__(self, api_key: str, secret_key: str, feed: DataFeed = DataFeed.IEX, **kwargs: Any):
        super().__init__()
        self.client = StockHistoricalDataClient(api_key, secret_key, **kwargs)
        self.feed = feed

    def retrieve_bars(
        self,
        *symbols: str,
        start: datetime | str | None = None,
        end: datetime | str | None = None,
        resolution: TimeFrame | None = None,
        adjustment: Adjustment = Adjustment.ALL,
    ):
        """Retrieve bar data for the given symbols.

        Args:
            *symbols: The symbols to retrieve bar data for.
            start (datetime, optional): The start time for the data. Defaults to None.
            end (datetime, optional): The end time for the data. Defaults to None.
            resolution (TimeFrame, optional): The resolution of the data. Defaults to None.
            adjustment (Adjustment, optional): The adjustment type. Defaults to Adjustment.ALL.
        """
        resolution = resolution or TimeFrame(amount=1, unit=TimeFrameUnit.Day)  # type: ignore
        req = StockBarsRequest(
            symbol_or_symbols=list(symbols), timeframe=resolution, start=start, end=end, adjustment=adjustment, feed=self.feed # type: ignore
        )
        res = self.client.get_stock_bars(req)
        assert isinstance(res, BarSet)
        freq = str(resolution)
        self._process_bars(res.data, freq, AssetClass.US_EQUITY)

    def retrieve_trades(self, *symbols: str, start: datetime | str | None = None, end: datetime | str | None = None):
        """Retrieve trade data for the given symbols.

        Args:
            *symbols: The symbols to retrieve trade data for.
            start (datetime, optional): The start time for the data. Defaults to None.
            end (datetime, optional): The end time for the data. Defaults to None.
        """
        req = StockTradesRequest(symbol_or_symbols=list(symbols), start=start, end=end, feed=self.feed)  # type: ignore
        res = self.client.get_stock_trades(req)
        assert isinstance(res, TradeSet)
        self._process_trades(res.data, AssetClass.US_EQUITY)

    def retrieve_quotes(self, *symbols: str, start: datetime | str | None = None, end: datetime | str | None = None):
        """Retrieve quote data for the given symbols.

        Args:
            *symbols (str): The symbols to retrieve quote data for.
            start (datetime, optional): The start time for the data. Defaults to None.
            end (datetime, optional): The end time for the data. Defaults to None.
        """
        req = StockQuotesRequest(symbol_or_symbols=list(symbols), start=start, end=end, feed=self.feed)  # type: ignore
        res = self.client.get_stock_quotes(req)
        assert isinstance(res, QuoteSet)
        self._process_quotes(res.data, AssetClass.US_EQUITY)


class AlpacaHistoricCryptoFeed(_AlpacaHistoricFeed):
    """Get historic cryptocurrency prices from Alpaca.

    Support for bars and trades.

    Args:
        api_key (str): The API key for Alpaca.
        secret_key (str): The secret key for Alpaca.
        **kwargs: Additional keyword arguments.
    """

    def __init__(self, api_key: str, secret_key: str, **kwargs: Any):
        super().__init__()
        self.client = CryptoHistoricalDataClient(api_key, secret_key, **kwargs)

    def retrieve_bars(
        self,
        *symbols: str,
        start: datetime | str | None = None,
        end: datetime | str | None = None,
        resolution: TimeFrame | None = None,
    ):
        """Retrieve bar data for the given symbols.

        Args:
            *symbols: The symbols to retrieve bar data for.
            start (datetime, optional): The start time for the data. Defaults to None.
            end (datetime, optional): The end time for the data. Defaults to None.
            resolution (TimeFrame, optional): The resolution of the data. Defaults to None.
        """
        resolution = resolution or TimeFrame(amount=1, unit=TimeFrameUnit.Day)  # type: ignore
        req = CryptoBarsRequest(symbol_or_symbols=list(symbols), timeframe=resolution, start=start, end=end)  # type: ignore
        res = self.client.get_crypto_bars(req)
        assert isinstance(res, BarSet)
        freq = str(resolution)
        self._process_bars(res.data, freq, AssetClass.CRYPTO)

    def retrieve_trades(self, *symbols: str, start: datetime | str | None = None, end: datetime | str | None = None):
        """Retrieve trade data for the given symbols.

        Args:
            *symbols: The symbols to retrieve trade data for.
            start (datetime, optional): The start time for the data. Defaults to None.
            end (datetime, optional): The end time for the data. Defaults to None.
        """
        req = CryptoTradesRequest(symbol_or_symbols=list(symbols), start=start, end=end)  # type: ignore
        res = self.client.get_crypto_trades(req)
        assert isinstance(res, TradeSet)
        self._process_trades(res.data, AssetClass.CRYPTO)


class AlpacaBroker(LiveBroker):
    """Broker implementation for live and paper trading using the Alpaca trading API.
    This broker supports US equities, options, and crypto trading.
    It requires an Alpaca API key and secret key.
    """

    def __init__(self, api_key: str, secret_key: str) -> None:
        super().__init__()
        self.__client = TradingClient(api_key, secret_key)

    def _sync_orders(self):
        orders = []
        request = GetOrdersRequest(status=QueryOrderStatus.OPEN)
        alpaca_orders: list[AOrder] = self.__client.get_orders(request)  # type: ignore
        for alpaca_order in alpaca_orders:
            asset = _get_asset(alpaca_order.symbol, alpaca_order.asset_class)  # type: ignore
            id = str(alpaca_order.id)
            tif = "GTC" if alpaca_order.time_in_force == TimeInForce.GTC else "DAY"
            if alpaca_order.side == OrderSide.SELL:
                order = self._sell_order(id, asset, alpaca_order.qty,alpaca_order.limit_price,alpaca_order.filled_qty, tif)  # type: ignore
            else:
                order = self._buy_order(id, asset, alpaca_order.qty, alpaca_order.limit_price,alpaca_order.filled_qty, tif)  # type: ignore

            orders.append(order)

        return orders

    def _sync_positions(self):
        positions = Portfolio()
        open_pos: list[APosition] = self.__client.get_all_positions()  # type: ignore

        for p in open_pos:
            size = Decimal(p.qty)
            if p.side == PositionSide.SHORT:
                size = -size
            new_pos = Position(size, float(p.avg_entry_price), float(p.current_price or "nan"))
            asset = _get_asset(p.symbol, p.asset_class)
            positions[asset] = new_pos
        return positions

    def _get_account(self) -> Account:
        account = Account()
        acc: TradeAccount = self.__client.get_account()  # type: ignore
        if acc.buying_power:
            account.buying_power = Amount(USD, float(acc.buying_power))
        if acc.cash:
            account.cash = Wallet(Amount(USD, float(acc.cash)))

        account.portfolio = self._sync_positions()
        account.orders = self._sync_orders()
        return account

    def _cancel_order(self, order: Order):
        self.__client.cancel_order_by_id(order.id)

    def _update_order(self, order: Order):
        req = ReplaceOrderRequest(qty=int(abs(float(order.size))), limit_price=order.limit)
        result = self.__client.replace_order_by_id(order.id, req)
        logger.info("result update order oder=%s result=%s", order, result)

    def _place_order(self, order: Order):
        req = self._get_order_request(order)
        result = self.__client.submit_order(req)
        logger.info("result place order oder=%s result=%s", order, result)

    def _get_order_request(self, order: Order) -> LimitOrderRequest:
        side = OrderSide.BUY if order.is_buy else OrderSide.SELL
        return LimitOrderRequest(
            symbol=order.asset.symbol,
            qty=abs(float(order.size)),
            side=side,
            limit_price=order.limit,
            time_in_force=TimeInForce.GTC if order.tif == "GTC" else TimeInForce.DAY,
        )
