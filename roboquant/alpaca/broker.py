import logging
from decimal import Decimal
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import AssetClass, OrderSide, QueryOrderStatus, TimeInForce
from alpaca.trading.models import TradeAccount
from alpaca.trading.models import Position as APosition
from alpaca.trading.models import Order as AOrder

from alpaca.trading.requests import GetOrdersRequest, LimitOrderRequest, ReplaceOrderRequest
from roboquant.account import Account, Position
from roboquant.asset import Asset, Crypto, Option, Stock
from roboquant.brokers.broker import LiveBroker, Order
from roboquant.monetary import Wallet, Amount, USD

logger = logging.getLogger(__name__)


class AlpacaBroker(LiveBroker):
    """Alpaca Broker implementation for live trading"""

    def __init__(self, api_key: str, secret_key: str) -> None:
        super().__init__()
        self.__client = TradingClient(api_key, secret_key)

    def _get_asset(self, symbol: str, asset_class: AssetClass) -> Asset:
        match asset_class:
            case AssetClass.US_EQUITY:
                return Stock(symbol)
            case AssetClass.US_OPTION:
                return Option(symbol)
            case AssetClass.CRYPTO:
                return Crypto.from_symbol(symbol)

    def _sync_orders(self):
        orders: list[Order] = []
        request = GetOrdersRequest(status=QueryOrderStatus.OPEN)
        alpaca_orders: list[AOrder] = self.__client.get_orders(request)  # type: ignore
        for alpaca_order in alpaca_orders:
            alpaca_order.status
            asset = self._get_asset(alpaca_order.symbol, alpaca_order.asset_class)  # type: ignore
            order = Order(
                asset,
                Decimal(alpaca_order.qty),  # type: ignore
                float(alpaca_order.limit_price),  # type: ignore
            )
            order.fill = Decimal(alpaca_order.filled_qty)  # type: ignore
            order.id = str(alpaca_order.id)
            orders.append(order)

        return orders

    def _sync_positions(self):
        positions: dict[Asset, Position] = {}
        open_pos: list[APosition] = self.__client.get_all_positions()  # type: ignore

        for p in open_pos:
            size = Decimal(p.qty)
            if p.side == "short":
                size = -size
            new_pos = Position(size, float(p.avg_entry_price), float(p.current_price or "nan"))
            asset = self._get_asset(p.symbol, p.asset_class)
            positions[asset] = new_pos
        return positions

    def _get_account(self) -> Account:
        account = Account()
        acc: TradeAccount = self.__client.get_account()  # type: ignore
        if acc.buying_power:
            account.buying_power = Amount(USD, float(acc.buying_power))
        if acc.cash:
            account.cash = Wallet(Amount(USD, float(acc.cash)))

        account.positions = self._sync_positions()
        account.orders = self._sync_orders()
        return account

    def _cancel_order(self, order: Order):
        assert order.id is not None
        self.__client.cancel_order_by_id(order.id)

    def _update_order(self, order: Order):
        assert order.id is not None
        if order.gtd:
            logger.warning("no support for GTD type of orders, ignoring gtd=%s", order.gtd)
        req = ReplaceOrderRequest(qty=int(abs(float(order.size))), limit_price=order.limit)
        self.__client.replace_order_by_id(order.id, req)

    def _place_order(self, order: Order):
        if order.gtd:
            logger.warning("no support for GTD type of orders, ignoring gtd=%s", order.gtd)
        req = self._get_order_request(order)
        self.__client.submit_order(req)

    def _get_order_request(self, order: Order):
        side = OrderSide.BUY if order.is_buy else OrderSide.SELL
        return LimitOrderRequest(
            symbol=order.asset.symbol,
            qty=abs(float(order.size)),
            side=side,
            limit_price=order.limit,
            time_in_force=TimeInForce.GTC,
        )

