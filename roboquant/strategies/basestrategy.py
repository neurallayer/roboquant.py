from abc import abstractmethod
from datetime import timedelta
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
    """Base version of strategy that contains several methods to make it easier to create orders.
    """

    def __init__(self) -> None:
        super().__init__()
        self.order_value_perc = 0.1
        self.buy_price = "DEFAULT"
        self.sell_price = "DEFAULT"
        self.fractional_order_digits = 0
        self.cancel_existing_orders = True

        self._orders: list[Order]
        self._order_value: float
        self._remaining_buying_power: float
        self._account: Account
        self._event: Event

    def create_orders(self, event: Event, account: Account) -> list[Order]:
        self._orders = []
        self._remaining_buying_power = round(account.buying_power.value, 2)
        self._account = account
        self._event = event
        self._order_value = round(account.equity_value() * self.order_value_perc, 2)

        self.process(event, account)
        return self._orders

    def _get_size(self, asset: Asset, limit: float) -> Decimal:
        if self._order_value < 0.1:
            return Decimal()
        value_one = asset.contract_amount(Decimal(1), limit).convert(self._account.base_currency, self._account.last_update)
        return round(Decimal(self._order_value / value_one), self.fractional_order_digits)

    def add_buy_order(self, asset: Asset, limit: float | None = None):
        if limit := limit or self._get_limit(asset, True):
            if size := self._get_size(asset, limit):
                order = Order(asset, size, limit)
                return self.add_order(order)
        return False

    def add_exit_order(self, asset: Asset, limit: float | None = None):
        if size := -self._account.get_position_size(asset):
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
        limit_price = self._event.get_price(asset, price_type)
        return round(limit_price, 2) if limit_price else None

    def add_order(self, order: Order) -> bool:
        """Add an order if there is enough remaining buying power"""
        bp = self._account.required_buying_power(order).value
        if logger.isEnabledFor(level=logging.INFO):
            logger.info(
                "order=%s required=%s available=%s max_order_value=%s default_price=%s",
                order,
                bp,
                self._remaining_buying_power,
                self._order_value,
                self._event.get_price(order.asset),
            )
        if bp and bp > self._remaining_buying_power:
            logger.info("not enough buying power remaining")
            return False

        self._remaining_buying_power -= bp
        if self.cancel_existing_orders:
            self.cancel_open_orders(order.asset)
        self._orders.append(order)
        return True

    def cancel_old_orders(self, older_than: timedelta):
        for order in self._account.orders:
            if order.created_at + older_than < self._event.time:
                self.cancel_order(order)

    def cancel_open_orders(self, *assets):
        for order in self._account.orders:
            if not assets or order.asset in assets:
                self.cancel_order(order)

    def cancel_order(self, order: Order):
        self._orders.append(order.cancel())

    def modify_order(self, order: Order, size: float | None = None, limit: float | None = None):
        modify_order = order.modify(size=size, limit=limit)
        self._orders.append(modify_order)

    @abstractmethod
    def process(self, event: Event, account: Account):
        """Implement this method"""
        ...
