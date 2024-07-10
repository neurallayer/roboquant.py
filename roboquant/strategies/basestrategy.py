from abc import abstractmethod
from decimal import Decimal
import logging

from roboquant.asset import Asset
from roboquant.event import Event
from roboquant.order import Order
from roboquant.strategies.strategy import Strategy
from roboquant.account import Account


logger = logging.getLogger(__name__)


class BaseStrategy(Strategy):
    # pylint: disable=too-many-instance-attributes
    """Simplified version of strategy where you only have to implement the logic to generate signals and
    can leave the logic for orders to a trader.
    """

    def __init__(self) -> None:
        super().__init__()
        self.orders: list[Order]
        self.buying_power: float
        self.account: Account
        self.event: Event
        self.order_value: float
        self.order_value_perc = 0.1
        self.buy_price = "DEFAULT"
        self.sell_price = "DEFAULT"
        self.fractional_order_digits = 0
        self.cancel_existing_orders = True

    def create_orders(self, event: Event, account: Account) -> list[Order]:
        self.orders = []
        self.buying_power = round(account.buying_power.value, 2)
        self.account = account
        self.event = event
        self.order_value = round(account.equity_value() * self.order_value_perc, 2)

        self.process(event, account)
        return self.orders

    def _get_size(self, asset: Asset, limit: float) -> Decimal:
        if self.order_value < 0.1:
            return Decimal()
        value_one = asset.contract_amount(Decimal(1), limit).convert(self.account.base_currency, self.account.last_update)
        return round(Decimal(self.order_value / value_one), self.fractional_order_digits)

    def _required_buyingpower(self, order: Order) -> float:
        """How much buying power is required for the order"""
        pos_size = self.account.get_position_size(order.asset)
        if abs(pos_size + order.size) < abs(pos_size):
            return 0.0
        return abs(self.account.convert(order.amount()))

    def add_buy_order(self, asset: Asset, limit: float | None = None):
        if limit := limit or self._get_limit(asset, True):
            if size := self._get_size(asset, limit):
                order = Order(asset, size, limit)
                return self.add_order(order)
        return False

    def add_exit_order(self, asset: Asset, limit: float | None = None):
        if size := -self.account.get_position_size(asset):
            if limit := limit or self._get_limit(asset, size > 0):
                order = Order(asset, size, limit)
                return self.add_order(order)
        return False

    def add_sell_order(self, asset: Asset, limit: float | None = None):
        if limit := limit or self._get_limit(asset, False):
            if size := self._get_size(asset, limit) * -1:
                order = Order(asset, size, limit)
                return self.add_order(order)
        return False

    def _get_limit(self, asset: Asset, is_buy: bool) -> float | None:
        price_type = self.buy_price if is_buy else self.sell_price
        limit_price = self.event.get_price(asset, price_type)
        return round(limit_price, 2) if limit_price else None

    def add_order(self, order: Order) -> bool:
        """Add an order if there is enough remaining buying power"""
        bp = self._required_buyingpower(order)
        if logger.isEnabledFor(level=logging.INFO):
            logger.info(
                "order=%s required=%s available=%s max_order_value=%s default_price=%s",
                order,
                bp,
                self.buying_power,
                self.order_value,
                self.event.get_price(order.asset),
            )
        if bp and bp > self.buying_power:
            logger.info("not enough buying power remaining")
            return False

        self.buying_power -= bp
        if self.cancel_existing_orders:
            self.cancel_open_orders(order.asset)
        self.orders.append(order)
        return True

    def cancel_open_orders(self, *assets):
        for order in self.account.orders:
            if not assets or order.asset in assets:
                self.cancel_order(order)

    def cancel_order(self, order: Order):
        self.orders.append(order.cancel())

    def modify_order(self, order: Order, size: float | None = None, limit: float | None = None):
        modify_order = order.modify(size=size, limit=limit)
        self.orders.append(modify_order)

    @abstractmethod
    def process(self, event: Event, account: Account):
        """Implement this method"""
        ...
