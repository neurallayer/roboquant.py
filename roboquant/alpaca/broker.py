import logging
import time
from decimal import Decimal
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import AssetClass, OrderSide, QueryOrderStatus, TimeInForce
from alpaca.trading.models import TradeAccount
from alpaca.trading.models import Position as APosition
from alpaca.trading.models import Order as AOrder

from alpaca.trading.requests import GetOrdersRequest, LimitOrderRequest, ReplaceOrderRequest
from roboquant.account import Account, Position
from roboquant.asset import Asset, Crypto, Option, Stock
from roboquant.event import Event
from roboquant.brokers.broker import LiveBroker
from roboquant.order import Order
from roboquant.monetary import Wallet, Amount, USD

logger = logging.getLogger(__name__)


class AlpacaBroker(LiveBroker):
    """Alpaca Broker implementation for live trading"""

    def __init__(self, api_key: str, secret_key: str) -> None:
        super().__init__()
        self.__account: Account = Account()
        self.__client = TradingClient(api_key, secret_key)
        self.price_type = "DEFAULT"
        self.sleep_after_cancel = 0.0

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
            asset = self._get_asset(alpaca_order.symbol, alpaca_order.asset_class)  # type: ignore
            order = Order(
                asset,
                Decimal(alpaca_order.qty),  # type: ignore
                float(alpaca_order.limit_price),  # type: ignore
            )
            order.fill = Decimal(alpaca_order.filled_qty)  # type: ignore
            order.id = str(alpaca_order.id)
            orders.append(order)

        self.__account.orders = orders

    def _sync_positions(self):
        open_pos: list[APosition] = self.__client.get_all_positions()  # type: ignore
        self.__account.positions.clear()
        for p in open_pos:
            size = Decimal(p.qty)
            if p.side == "short":
                size = -size
            new_pos = Position(size, float(p.avg_entry_price), float(p.current_price or "nan"))
            asset = self._get_asset(p.symbol, p.asset_class)
            self.__account.positions[asset] = new_pos

    def sync(self, event: Event | None = None) -> Account:
        now = self.guard(event)

        client = self.__client
        acc: TradeAccount = client.get_account()  # type: ignore
        if acc.buying_power:
            self.__account.buying_power = Amount(USD, float(acc.buying_power))
        if acc.cash:
            self.__account.cash = Wallet(Amount(USD, float(acc.cash)))

        self.__account.last_update = now

        self._sync_positions()
        self._sync_orders()
        return self.__account

    def place_orders(self, orders):

        for order in orders:

            if order.gtd:
                logger.warning("no support for GTD type of orders, ignoring gtd=%s", order.gtd)

            if order.is_cancellation:
                assert order.id is not None, "can only cancel orders with an id"
                self.__client.cancel_order_by_id(order.id)
                if self.sleep_after_cancel:
                    time.sleep(self.sleep_after_cancel)
            else:
                if order.id is None:
                    req = self._get_order_request(order)
                    alpaca_order = self.__client.submit_order(req)
                    order.id = str(alpaca_order.id)  # type: ignore
                    self.__account.orders.append(order)
                else:
                    req = self._get_replace_request(order)
                    self.__client.replace_order_by_id(order.id, req)

    def _get_order_request(self, order: Order):
        side = OrderSide.BUY if order.is_buy else OrderSide.SELL
        return LimitOrderRequest(
            symbol=order.asset.symbol,
            qty=abs(float(order.size)),
            side=side,
            limit_price=order.limit,
            time_in_force=TimeInForce.GTC,
        )

    def _get_replace_request(self, order: Order):
        result = ReplaceOrderRequest(qty=int(abs(float(order.size))), limit_price=order.limit)
        return result
