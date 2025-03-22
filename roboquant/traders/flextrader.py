from datetime import datetime, timedelta
import logging
from decimal import Decimal
from enum import Flag, auto
import random
from typing import Any

from roboquant.asset import Asset
from roboquant.event import Event
from roboquant.order import Order
from roboquant.signal import Signal
from .trader import Trader
from ..account import Account
from ..event import PriceItem

logger = logging.getLogger(__name__)


class _PositionChange(Flag):
    """representing the four types of changes to a portfolio"""

    ENTRY_LONG = auto()
    ENTRY_SHORT = auto()
    EXIT_LONG = auto()
    EXIT_SHORT = auto()

    _ENTRY = ENTRY_LONG | ENTRY_SHORT
    _EXIT = EXIT_LONG | EXIT_SHORT

    @property
    def is_entry(self):
        """Return True is the status is open, False otherwise"""
        return self in _PositionChange._ENTRY

    @property
    def is_exit(self):
        """Return True is the status is closed, False otherwise"""
        return self in _PositionChange._EXIT

    @staticmethod
    def get_change(is_buy: bool, pos_size: Decimal) -> "_PositionChange":
        """Determine the kind of change a certain action would have on the position"""
        if pos_size.is_zero():
            return _PositionChange.ENTRY_LONG if is_buy else _PositionChange.ENTRY_SHORT
        if pos_size > 0:
            return _PositionChange.ENTRY_LONG if is_buy else _PositionChange.EXIT_LONG

        return _PositionChange.EXIT_SHORT if is_buy else _PositionChange.ENTRY_SHORT

    def __repr__(self) -> str:
        return self.name.split(".")[-1]  # type: ignore


class _Context:
    def __init__(self, time: datetime) -> None:
        self.time = time.replace(tzinfo=None)  # Allow for nicer printing

    def log_received(self, **kwargs):
        if logger.isEnabledFor(logging.INFO):
            extra = " ".join(f"{k}={v}" for k, v in kwargs.items())
            logger.info(
                "==> %s received %s",
                self.time,
                extra
            )

    def log_orders(self, orders):
        """Log an exit due to to a signal being converted into an order"""
        if logger.isEnabledFor(logging.INFO):
            logger.info(
                "<== %s converter signal into order(s) %s",
                self.time,
                orders,
            )

    def log_rule(self, rule: str, **kwargs: Any):
        """Log an exit due to a signal being discarded by a triggered rule"""
        if logger.isEnabledFor(logging.INFO):
            extra = " ".join(f"{k}={v}" for k, v in kwargs.items())
            logger.info(
                "<== %s discarded signal because of '%s' rule %s",
                self.time,
                rule,
                extra,
            )


class FlexTrader(Trader):
    """Implementation of a Trader that has configurable rules to determine which signals are converted into orders.
    This implementation will not generate orders if there is not a price in the event for the underlying asset.

    The configurable parameters include:

    - one_order_only: don't create new orders for an asset if there is already an open orders for that same asset
    - size_fractions: enable fractional order sizes (if size_fractions is larger than 0), default is 0
    - safety_margin_perc: the safety margin as percentage of equity that should remain available (to avoid margin calls),
    default is 0.05 (5%)
    - max_position_perc: the max percentage of the equity to allocate to a single position, default is 0.1 (10%)
    - max_order_perc: the max percentage of the equity to allocate to a new order, default is 0.05 (5%)
    - min_order_perc: the min percentage of the equity to allocate to a new order, default is 0.02 (2%)
    - shorting: allow orders that could result in a short position, default is false
    - price_type: the price type to use when determining order value, for example "CLOSE". Default is "DEFAULT"
    - shuffle_signals: shuffle the signals before processing them, default is false
    - valid_for: the time delta for which the order is valid, default is 3 days

    It might be sometimes challenging to understand why a signal isn't converted into an order. The flex-trader logs
    at INFO level when certain rules have been fired. Enable higher logging:
    ```
        logging.basicConfig()
        logging.getLogger("roboquant.traders.flextrader").setLevel(logging.INFO)
    ```
    """

    def __init__(
        self,
        one_order_only: bool = True,
        size_fractions: int = 0,
        safety_margin_perc: float = 0.05,
        shorting: bool = False,
        max_order_perc: float = 0.05,
        min_order_perc: float = 0.02,
        max_position_perc: float = 0.1,
        price_type: str = "DEFAULT",
        shuffle_signals: bool = False,
    ) -> None:
        super().__init__()
        self.one_order_only = one_order_only
        self.size_digits: int = size_fractions
        self.safety_margin_perc: float = safety_margin_perc
        self.shorting = shorting
        self.max_order_perc = max_order_perc
        self.min_order_perc = min_order_perc
        self.max_position_perc = max_position_perc
        self.price_type = price_type
        self.shuffle_signals = shuffle_signals
        self.valid_for: timedelta | None = timedelta(days=3)

    def _get_order_size(self, rating: float, contract_price: float, max_order_value: float) -> Decimal:
        """Return the order size"""
        size = Decimal(rating * max_order_value / contract_price)
        rounded_size = round(size, self.size_digits)
        return rounded_size

    def create_orders(self, signals: list[Signal], event: Event, account: Account) -> list[Order]:
        # pylint: disable=too-many-branches,too-many-statements,too-many-locals
        if not signals:
            return []

        if self.shuffle_signals:
            random.shuffle(signals)

        orders: list[Order] = []
        equity = account.equity_value()
        max_order_value = equity * self.max_order_perc
        min_order_value = equity * self.min_order_perc
        max_pos_value = equity * self.max_position_perc
        available = account.buying_power.value - self.safety_margin_perc * equity
        order_assets = {order.asset for order in account.orders}
        ctx = _Context(event.time)

        for signal in signals:
            asset = signal.asset
            pos_size = account.get_position_size(asset)
            change = _PositionChange.get_change(signal.is_buy, pos_size)

            ctx.log_received(signal=signal, position=pos_size, available=available)

            # logger.info("==> received signal available=%s signal=%s pos=%s change=%s", available, signal, pos_size, change)

            if self.one_order_only and asset in order_assets:
                ctx.log_rule("one order only")
                continue

            item = event.price_items.get(asset)
            if item is None:
                ctx.log_rule("no known price")
                continue

            price = item.price(self.price_type)

            if not self.shorting and change == _PositionChange.ENTRY_SHORT:
                ctx.log_rule("no shorting")
                continue

            if change.is_exit:
                # Closing orders don't require or use buying power
                if not signal.is_exit:
                    ctx.log_rule("no exit signal")
                    continue

                rounded_size = round(-pos_size * abs(Decimal(signal.rating)), self.size_digits)
                if rounded_size.is_zero():
                    ctx.log_rule("cannot exit with order size zero")
                    continue
                new_orders = self._get_orders(asset, rounded_size, item, signal, event.time)
                orders += new_orders
            else:
                if available < 0:
                    ctx.log_rule("no more available buying power")
                    continue

                if not signal.is_entry:
                    ctx.log_rule("no entry signal")
                    continue

                if available < min_order_value:
                    ctx.log_rule("available buying power below minimum order value")
                    continue

                available_order_value = min(available, max_order_value, max_pos_value - abs(account.position_value(asset)))
                if available_order_value < min_order_value:
                    ctx.log_rule("calculated available order value below minimum order value")
                    continue

                contract_price = account.contract_value(asset, Decimal(1), price)
                order_size = self._get_order_size(signal.rating, contract_price, available_order_value)

                if order_size.is_zero():
                    ctx.log_rule("calculated order size is zero")
                    continue

                order_value = abs(account.contract_value(asset, order_size, price))
                if abs(order_value) > available:
                    ctx.log_rule(
                        "order value above available buying power",
                        order_value=order_value,
                        available=available,
                    )
                    continue
                if abs(order_value) < min_order_value:
                    ctx.log_rule(
                        "order value below minimum order value",
                        order_value=order_value,
                        min_order_value=min_order_value,
                    )
                    continue

                new_orders = self._get_orders(asset, order_size, item, signal, event.time)
                if new_orders:
                    orders += new_orders
                    available -= order_value

        return orders

    def _get_orders(self, asset: Asset, size: Decimal, item: PriceItem, signal: Signal, time: datetime) -> list[Order]:
        # pylint: disable=unused-argument
        """Return zero or more orders for the provided asset and size.
        The default implementation:
        - creates a single order
        - with the limit price being the `self.price_type` rounded to two decimals
        - the gtd set to the time of the event + `self.valid_for`

        Overwrite this method if you want to implement different logic.
        """
        gtd = None if not self.valid_for else time + self.valid_for
        price = item.price(self.price_type)
        limit = round(price, 2)
        result = [Order(asset, size, limit, gtd)]
        logger.info("<== %s converted signal into new order(s) %s", time.replace(tzinfo=None), result)
        return result

    def __str__(self) -> str:
        attrs = " ".join([f"{k}={v}" for k, v in self.__dict__.items() if not k.startswith("_")])
        return f"FlexTrader({attrs})"
