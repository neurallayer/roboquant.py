from abc import abstractmethod
from decimal import Decimal
import logging

from roboquant.event import Event
from roboquant.order import Order, OrderType
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
            return -pos_size

        return size if order_type.is_buy else size * -1

    def _required_buyingpower(self, symbol: str, size: Decimal, limit: float) -> float:
        pos_size = self.account.get_position_size(symbol)
        if abs(pos_size - size) < abs(pos_size):
            return 0.0
        return abs(self.account.contract_value(symbol, limit, size))

    def add_buy_order(self, symbol: str, limit: float | None = None):
        if limit := limit or self._get_limit(symbol, OrderType.BUY):
            if size := self._get_size(symbol, OrderType.BUY, limit):
                return self.add_order(symbol, size, limit)
        return False

    def add_exit_order(self, symbol: str, limit: float | None = None):
        if limit := limit or self._get_limit(symbol, OrderType.BUY):
            if size := - self.account.get_position_size(symbol):
                return self.add_order(symbol, size, limit)
        return False

    def add_sell_order(self, symbol: str, limit: float | None = None):
        if limit := limit or self._get_limit(symbol, OrderType.SELL):
            if size := self._get_size(symbol, OrderType.SELL, limit):
                return self.add_order(symbol, size, limit)
        return False

    def _get_limit(self, symbol: str, order_type: OrderType) -> float | None:
        price_type = self.buy_price if order_type.is_buy else self.sell_price
        limit_price = self.event.get_price(symbol, price_type)
        return round(limit_price, 2) if limit_price else None

    def add_order(self, symbol: str, size: Decimal, limit: float) -> bool:
        if not self.short_selling and size < 0 and self.account.get_position_size(symbol) <= 0:
            logger.info("no short selling allowed")
            return False

        bp = self._required_buyingpower(symbol, size, limit)
        if bp and bp > self.buying_power:
            logger.info("not enough buying power remaining")
            return False

        self.buying_power -= bp
        order = Order(symbol, size, limit)
        self.orders.append(order)
        return True

    def cancel_open_orders(self, *symbols):
        for order in self.account.orders:
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
