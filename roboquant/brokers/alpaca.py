from decimal import Decimal
from typing import override

from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, PositionSide, QueryOrderStatus, TimeInForce
from alpaca.trading.models import Order as AOrder
from alpaca.trading.models import Position as APosition
from alpaca.trading.models import TradeAccount
from alpaca.trading.requests import GetOrdersRequest, LimitOrderRequest, MarketOrderRequest, ReplaceOrderRequest

from roboquant.brokers.livebroker import LiveBroker
from roboquant.common.account import Account
from roboquant.common.monetary import USD, Amount, Wallet
from roboquant.common.order import Order
from roboquant.common.position import Position
from roboquant.common.timeframe import utcnow
from roboquant.feeds.alpaca import _get_asset, logger


class AlpacaBroker(LiveBroker):
    """
    Broker implementation for live and paper trading using the Alpaca trading API.
    This broker supports US equities, options, and crypto trading.
    It requires an Alpaca API key and secret key.
    """

    def __init__(self, api_key: str, secret_key: str) -> None:
        super().__init__()
        self.__client = TradingClient(api_key, secret_key)

    def __sync_orders(self):
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

    def __sync_positions(self):
        positions = []
        open_pos: list[APosition] = self.__client.get_all_positions()  # type: ignore

        for p in open_pos:
            size = Decimal(p.qty)
            if p.side == PositionSide.SHORT:
                size = -size
            asset = _get_asset(p.symbol, p.asset_class)
            info = {"asset_id" : p.asset_id}
            price = float(p.current_price or p.lastday_price or "nan")
            new_pos = Position(asset, size, float(p.avg_entry_price), price, info=info)
            positions.append(new_pos)
        return positions

    @override
    def _get_account(self) -> Account:
        acc: TradeAccount = self.__client.get_account()  # type: ignore
        return Account(
            buying_power=Amount(USD, float(acc.buying_power or 0.0)),
            positions= self.__sync_positions(),
            orders = self.__sync_orders(),
            last_update=utcnow(),
            cash = Wallet(Amount(USD, float(acc.cash or 0.0))),
            trades = []
        )

    @override
    def _cancel_order(self, order: Order):
        self.__client.cancel_order_by_id(order.id)

    @override
    def _update_order(self, order: Order):
        req = ReplaceOrderRequest(qty=int(abs(float(order.size))), limit_price=order.limit)
        result = self.__client.replace_order_by_id(order.id, req)
        logger.info("result update order oder=%s result=%s", order, result)

    @override
    def _place_order(self, order: Order):
        req = self._get_order_request(order)
        result = self.__client.submit_order(req)
        logger.info("result place order oder=%s result=%s", order, result)

    def _get_order_request(self, order: Order) -> LimitOrderRequest | MarketOrderRequest:
        side = OrderSide.BUY if order.is_buy else OrderSide.SELL

        if order.is_mkt_order:
            return MarketOrderRequest(
                symbol=order.asset.symbol,
                qty=abs(float(order.size)),
                side=side,
                time_in_force=TimeInForce.GTC if order.tif == "GTC" else TimeInForce.DAY,
            )

        return LimitOrderRequest(
            symbol=order.asset.symbol,
            qty=abs(float(order.size)),
            side=side,
            limit_price=order.limit,
            time_in_force=TimeInForce.GTC if order.tif == "GTC" else TimeInForce.DAY,
        )
