from abc import abstractmethod
from datetime import datetime, timedelta
from decimal import Decimal
import logging

from roboquant.event import Event
from roboquant.order import Order, OrderType
from roboquant.strategies.strategy import Strategy
from roboquant.account import Account


logger = logging.getLogger(__name__)


class BaseStrategy(Strategy):
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
        self.order_valid_for = timedelta(days=3)
        self.fractional_order_digits = 0
        self.short_selling = False

    def create_orders(self, event: Event, account: Account) -> list[Order]:
        self.orders = []
        self.buying_power = account.buying_power
        self.account = account
        self.event = event
        self.order_value = account.equity() * self.order_value_perc

        self.process(event, account)
        return self.orders

    def _get_size(self, symbol: str, order_type: OrderType, limit: float) -> Decimal:
        value_one = self.account.contract_value(symbol, limit)
        size = round(Decimal(self.order_value / value_one), self.fractional_order_digits)

        pos_size = self.account.get_position_size(symbol)
        if order_type.is_closing(pos_size) and abs(size) > abs(pos_size):
            return - pos_size

        return size if order_type.is_buy else size * -1

    def _get_gtc(self) -> datetime:
        """return the Good Till Cancelled datetime to be used for an order"""
        return self.event.time + self.order_valid_for

    def _required_buyingpower(self, symbol: str, size: Decimal, limit: float) -> float:
        pos_size = self.account.get_position_size(symbol)
        if abs(pos_size - size) < abs(pos_size):
            return 0.0
        return abs(self.account.contract_value(symbol, limit, size))

    def add_buy_order(self, symbol: str, limit: float | None = None):
        self.add_order(symbol, OrderType.BUY, limit)

    def add_sell_order(self, symbol: str, limit: float | None = None):
        self.add_order(symbol, OrderType.SELL, limit)

    def get_limit_price(self, symbol: str, order_type: OrderType):
        price_type = self.buy_price if order_type.is_buy else self.sell_price
        return self.event.get_price(symbol, price_type)

    def add_order(self, symbol: str, order_type: OrderType, limit: float | None) -> bool:
        if not self.short_selling and order_type.is_sell and self.account.get_position_size(symbol) <= 0:
            logger.info("no short selling allowed")
            return False

        limit = limit or self.get_limit_price(symbol, order_type)
        if not limit:
            logger.info("couldn't determine limit for order")
            return False

        size = self._get_size(symbol, order_type, limit)
        if size:
            bp = self._required_buyingpower(symbol, size, limit)
            if bp and bp > self.buying_power:
                logger.info("not enough buying power remaining")
                return False

            self.buying_power -= bp
            order = Order(symbol, size, limit, self._get_gtc())
            self.orders.append(order)
            return True

        logger.info("size is zero")
        return False

    def cancel_open_orders(self, *symbols):
        for order in self.account.open_orders:
            if not symbols or order.symbol in symbols:
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
