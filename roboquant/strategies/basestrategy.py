from abc import abstractmethod
from decimal import Decimal
import logging

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
        self.buying_power = account.buying_power
        self.account = account
        self.event = event
        self.order_value = account.equity() * self.order_value_perc

        self.process(event, account)
        return self.orders

    def _get_size(self, symbol: str, limit: float) -> Decimal:
        value_one = self.account.contract_value(symbol, limit)
        return round(Decimal(self.order_value / value_one), self.fractional_order_digits)

    def _required_buyingpower(self, symbol: str, size: Decimal, limit: float) -> float:
        pos_size = self.account.get_position_size(symbol)
        if abs(pos_size + size) < abs(pos_size):
            return 0.0
        return abs(self.account.contract_value(symbol, limit, size))

    def add_buy_order(self, symbol: str, limit: float | None = None):
        if limit := limit or self._get_limit(symbol, True):
            if size := self._get_size(symbol, limit):
                return self.add_order(symbol, size, limit)
        return False

    def add_exit_order(self, symbol: str, limit: float | None = None):
        if size := - self.account.get_position_size(symbol):
            if limit := limit or self._get_limit(symbol, size > 0):
                return self.add_order(symbol, size, limit)
        return False

    def add_sell_order(self, symbol: str, limit: float | None = None):
        if limit := limit or self._get_limit(symbol, False):
            if size := self._get_size(symbol, limit) * -1:
                return self.add_order(symbol, size, limit)
        return False

    def _get_limit(self, symbol: str, is_buy: bool) -> float | None:
        price_type = self.buy_price if is_buy else self.sell_price
        limit_price = self.event.get_price(symbol, price_type)
        return round(limit_price, 2) if limit_price else None

    def add_order(self, symbol: str, size: Decimal, limit: float) -> bool:
        bp = self._required_buyingpower(symbol, size, limit)
        logger.info("symbol=%s size=%s limit=%s required=%s available=%s", symbol, size, limit, bp, self.buying_power)
        if bp and bp > self.buying_power:
            logger.info("not enough buying power remaining")
            return False

        self.buying_power -= bp
        if self.cancel_existing_orders:
            self.cancel_open_orders(symbol)
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
