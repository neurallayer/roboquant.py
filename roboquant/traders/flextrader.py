import logging
from decimal import Decimal
from enum import Enum

from roboquant.event import Event
from roboquant.order import Order
from roboquant.signal import Signal
from .trader import Trader
from ..account import Account
from ..event import PriceItem

logger = logging.getLogger(__name__)


class _PositionChange(Enum):
    OPEN_LONG = 1
    OPEN_SHORT = 2
    CLOSE = 3
    INCREASE = 4

    @staticmethod
    def get_change(rating: float, pos_size: Decimal) -> "_PositionChange":
        """Determine the kind of change a certain rating has on the position"""
        if pos_size.is_zero():
            return _PositionChange.OPEN_LONG if rating > 0.0 else _PositionChange.OPEN_SHORT
        return _PositionChange.CLOSE if float(pos_size) * rating < 0.0 else _PositionChange.INCREASE


class FlexTrader(Trader):
    """Implementation of a Trader that has configurable rules to modify its behavior. This implementation will not
    generate orders if there is not a price in the event for the underlying symbol.

    The configurable parameters include:

    - one_order_only: don't create new orders for a symbol if there is already an open orders for that same symbol
    - size_fractions: enable fractional order sizes (if size_fractions is larger than 0), default is 0
    - min_buying_power: the minimal buying power that should remain available (to avoid margin calls), default is 0.0
    - increase_position: allow an order to potentially increase an open position size, default is False
    - max_order_perc: the max percentage of the equity to allocate to a new order, default is 0.05 (5%)
    - min_order_perc: the min percentage of the equity to allocate to a new order, default is 0.02 (2%)
    - shorting: allow orders that could result in a short position, default is false
    - price_type: the price type to use when determining order value

    """

    def __init__(
            self,
            one_order_only=True,
            size_fractions=0,
            min_buying_power=0.0,
            increase_position=False,
            shorting=False,
            max_order_perc=0.05,
            min_order_perc=0.02,
            price_type="DEFAULT",
    ) -> None:
        super().__init__()
        self.one_order_only = one_order_only
        self.size_digits: int = size_fractions
        self.min_buying_power: float = min_buying_power
        self.increase_position = increase_position
        self.shorting = shorting
        self.max_order_perc = max_order_perc
        self.min_order_perc = min_order_perc
        self.price_type = price_type

    def _get_order_size(self, rating: float, contract_price: float, max_order_value: float) -> Decimal:
        """Return the order size"""
        size = Decimal(rating * max_order_value / contract_price)
        rounded_size = round(size, self.size_digits)
        return rounded_size

    def create_orders(self, signals: dict[str, Signal], event: Event, account: Account) -> list[Order]:
        if not signals:
            return []

        orders: list[Order] = []
        buying_power = account.buying_power
        max_order_value = account.equity * self.max_order_perc
        min_order_value = account.equity * self.min_order_perc
        for symbol, signal in signals.items():

            if self.one_order_only and account.has_open_order(symbol):
                logger.debug("rating=%s for symbol=%s discarded because of one order rule", signal, symbol)
                continue

            item = event.price_items.get(symbol)
            if item is None:
                logger.debug("rating=%s for symbol=%s discarded because of no price available", signal, symbol)
                continue

            price = item.price(self.price_type)
            pos_size = account.get_position_size(symbol)

            change = _PositionChange.get_change(signal.rating, pos_size)
            if not self.shorting and change == _PositionChange.OPEN_SHORT:
                logger.debug("signal=%s for symbol=%s discarded because of shorting rule", signal, symbol)
                continue
            if not self.increase_position and change == _PositionChange.INCREASE:
                logger.debug("signal=%s for symbol=%s discarded because of increase position rule", signal, symbol)
                continue

            if change == _PositionChange.CLOSE:
                # Closing orders don't require or use buying power
                new_orders = self._get_orders(symbol, pos_size * -1, item, signal.rating)
                orders += new_orders
            else:
                contract_price = account.contract_value(symbol, Decimal(1), price)
                order_size = self._get_order_size(signal.rating, contract_price, max_order_value)

                order_value = abs(account.contract_value(symbol, order_size, price))
                if order_value > (buying_power - self.min_buying_power):
                    logger.debug("signal=%s for symbol=%s discarded because of insufficient buying power", signal, symbol)
                    continue
                if order_value < min_order_value:
                    logger.debug("signal=%s for symbol=%s discarded because of minimum order value", signal, symbol)
                    continue

                new_orders = self._get_orders(symbol, order_size, item, signal.rating)
                if new_orders:
                    orders += new_orders
                    buying_power -= order_value

        return orders

    def _get_orders(self, symbol: str, size: Decimal, item: PriceItem, rating: float) -> list[Order]:
        """Return zero or more orders for the provided symbol and size, default is a single a Market Order.

        Overwrite this method to create different order(s).
        """
        return [Order(symbol, size)]
