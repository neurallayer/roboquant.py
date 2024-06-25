
from datetime import datetime, timedelta
from decimal import Decimal
import logging

from roboquant.event import Event
from roboquant.order import Order, OrderType
from roboquant.account import Account


logger = logging.getLogger(__name__)


class FlexOrderManager:
    """Helps to create orders while adhereing to a number of rules
    """

    def __init__(self) -> None:
        super().__init__()
        self.order_value_perc = 0.1
        self.buy_price = "DEFAULT"
        self.sell_price = "DEFAULT"
        self.order_valid_for = timedelta(days=3)
        self.fractional_order_digits = 0
        self.short_selling = False

        # internal state objects
        self._orders: list[Order]
        self._buying_power: float
        self._account: Account
        self._event: Event
        self._order_value: float

    def next(self, event: Event, account: Account):
        self._orders = []
        self._buying_power = account.buying_power
        self._account = account
        self._event = event
        self._order_value = account.equity() * self.order_value_perc

    def _get_size(self, symbol: str, order_type: OrderType, limit: float) -> Decimal:
        value_one = self._account.contract_value(symbol, limit)
        size = round(Decimal(self._order_value / value_one), self.fractional_order_digits)

        pos_size = self._account.get_position_size(symbol)
        if order_type.is_closing(pos_size) and abs(size) > abs(pos_size):
            return - pos_size

        return size if order_type.is_buy else size * -1

    def _get_gtc(self) -> datetime:
        """return the Good Till Cancelled datetime to be used for an order"""
        return self._event.time + self.order_valid_for

    def _required_buyingpower(self, symbol: str, size: Decimal, limit: float) -> float:
        pos_size = self._account.get_position_size(symbol)
        if abs(pos_size - size) < abs(pos_size):
            return 0.0
        return abs(self._account.contract_value(symbol, limit, size))

    def add_buy_order(self, symbol: str, limit: float | None = None):
        self.add_order(symbol, OrderType.BUY, limit)

    def add_sell_order(self, symbol: str, limit: float | None = None):
        self.add_order(symbol, OrderType.SELL, limit)

    def get_limit_price(self, symbol: str, order_type: OrderType):
        price_type = self.buy_price if order_type.is_buy else self.sell_price
        return self._event.get_price(symbol, price_type)

    def add_order(self, symbol: str, order_type: OrderType, limit: float | None) -> bool:
        if not self.short_selling and order_type.is_sell and self._account.get_position_size(symbol) <= 0:
            logger.info("no short selling allowed")
            return False

        limit = limit or self.get_limit_price(symbol, order_type)
        if not limit:
            logger.info("couldn't determine limit for order")
            return False

        size = self._get_size(symbol, order_type, limit)
        if size:
            bp = self._required_buyingpower(symbol, size, limit)
            if bp and bp > self._buying_power:
                logger.info("not enough buying power remaining")
                return False

            self._buying_power -= bp
            order = Order(symbol, size, limit, self._get_gtc())
            self._orders.append(order)
            logger.info("added order %s", order)
            return True

        logger.info("size is zero")
        return False

    def get_orders(self):
        return self._orders

    def cancel_open_orders(self, *symbols):
        for order in self._account.open_orders:
            if not symbols or order.symbol in symbols:
                self.cancel_order(order)

    def cancel_order(self, order: Order):
        self._orders.append(order.cancel())

    def modify_order(self, order: Order, size: float | None = None, limit: float | None = None):
        modify_order = order.modify(size=size, limit=limit)
        self._orders.append(modify_order)
