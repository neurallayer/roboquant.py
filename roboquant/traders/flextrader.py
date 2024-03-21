from datetime import datetime, timedelta
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


def _log_rule(rule: str, signal: Signal, symbol: str, position: Decimal):
    if logger.isEnabledFor(logging.INFO):
        logger.info("signal=%s symbol=%s position=%s discarded because of %s", signal, symbol, position, rule)


class FlexTrader(Trader):
    """Implementation of a Trader that has configurable rules to modify which signals are converted into orders.
    This implementation will not generate orders if there is not a price in the event for the underlying symbol.

    The configurable parameters include:

    - one_order_only: don't create new orders for a symbol if there is already an open orders for that same symbol
    - size_fractions: enable fractional order sizes (if size_fractions is larger than 0), default is 0
    - min_buying_power: the minimal buying power that should remain available (to avoid margin calls), default is 0.0
    - increase_position: allow an order to potentially increase an open position size, default is False
    - max_order_perc: the max percentage of the equity to allocate to a new order, default is 0.05 (5%)
    - min_order_perc: the min percentage of the equity to allocate to a new order, default is 0.02 (2%)
    - shorting: allow orders that could result in a short position, default is false
    - price_type: the price type to use when determining order value

    It might be sometimes challenging to understand wby a signal isn't converted into an order. The flex-trader logs
    at INFO level when certain rules have been fired.

    """

    def __init__(
        self,
        one_order_only=True,
        size_fractions=0,
        min_buying_power_perc=0.05,
        increase_position=False,
        shorting=False,
        max_order_perc=0.05,
        min_order_perc=0.02,
        price_type="DEFAULT",
    ) -> None:
        super().__init__()
        self.one_order_only = one_order_only
        self.size_digits: int = size_fractions
        self.min_buying_power_perc: float = min_buying_power_perc
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
        # pylint: disable=too-many-branches,too-many-statements
        if not signals:
            return []

        orders: list[Order] = []
        equity = account.equity()
        max_order_value = equity * self.max_order_perc
        min_order_value = equity * self.min_order_perc
        available = account.buying_power - self.min_buying_power_perc * equity

        for symbol, signal in signals.items():
            pos_size = account.get_position_size(symbol)

            if self.one_order_only and account.has_open_order(symbol):
                _log_rule("one order only", signal, symbol, pos_size)
                continue

            item = event.price_items.get(symbol)
            if item is None:
                _log_rule("no price is available", signal, symbol, pos_size)
                continue

            price = item.price(self.price_type)
            change = _PositionChange.get_change(signal.rating, pos_size)

            if not self.shorting and change == _PositionChange.OPEN_SHORT:
                _log_rule("no shorting", signal, symbol, pos_size)
                continue
            if not self.increase_position and change == _PositionChange.INCREASE:
                _log_rule("no increase position sizing", signal, symbol, pos_size)
                continue

            if change == _PositionChange.CLOSE:
                # Closing orders don't require or use buying power
                if not signal.is_exit:
                    _log_rule("no exit signal", signal, symbol, pos_size)
                    continue
                new_orders = self._get_orders(symbol, pos_size * -1, item, signal.rating, event.time)
                orders += new_orders
            else:
                if not signal.is_entry:
                    _log_rule("no entry signal", signal, symbol, pos_size)
                    continue

                if available < min_order_value:
                    _log_rule("available buying power below minimum order value", signal, symbol, pos_size)
                    continue

                available_order_value = min(available, max_order_value)
                contract_price = account.contract_value(symbol, Decimal(1), price)
                order_size = self._get_order_size(signal.rating, contract_price, available_order_value)

                if order_size.is_zero():
                    _log_rule("calculated order size is zero", signal, symbol, pos_size)
                    continue

                order_value = abs(account.contract_value(symbol, order_size, price))
                if order_value > available:
                    _log_rule("order value above available buying power", signal, symbol, pos_size)
                    continue
                if order_value < min_order_value:
                    _log_rule("order value below minimum order value", signal, symbol, pos_size)
                    continue

                new_orders = self._get_orders(symbol, order_size, item, signal.rating, event.time)
                if new_orders:
                    orders += new_orders
                    available -= order_value

        return orders

    def _get_orders(self, symbol: str, size: Decimal, item: PriceItem, rating: float, time: datetime) -> list[Order]:
        # pylint: disable=unused-argument
        """Return zero or more orders for the provided symbol and size.

        Default is a MarketOrder. Overwrite this method to create different order types.
        """

        return [Order(symbol, size)]

    def __str__(self) -> str:
        attrs = " ".join([f"{k}={v}" for k, v in self.__dict__.items() if not k.startswith("_")])
        return f"FlexTrader({attrs})"


class FlexLimitOrderTrader(FlexTrader):
    """A FlexTrader version that returns a limit order"""

    def __init__(
        self,
        one_order_only=True,
        size_fractions=0,
        min_buying_power_perc=0.05,
        increase_position=False,
        shorting=False,
        max_order_perc=0.05,
        min_order_perc=0.02,
        price_type="DEFAULT",
        gtd=timedelta(days=3)
    ) -> None:
        super().__init__(
            one_order_only,
            size_fractions,
            min_buying_power_perc,
            increase_position,
            shorting,
            max_order_perc,
            min_order_perc,
            price_type,
        )
        self.gtd_timedelta = gtd

    def _get_orders(self, symbol: str, size: Decimal, item: PriceItem, rating: float, time: datetime) -> list[Order]:
        # pylint: disable=unused-argument
        """Return a single limit-order with the limit the current price a GTD with a configurable 3 days from now."""
        gtd = time + self.gtd_timedelta
        limit = item.price(self.price_type)
        return [Order(symbol, size, limit, gtd)]
