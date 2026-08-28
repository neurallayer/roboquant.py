from dataclasses import dataclass
from datetime import datetime
import logging
from decimal import Decimal
from enum import Flag, auto
import random
from typing import Any, Literal, override

from roboquant.common.asset import Asset
from roboquant.common.event import Event
from roboquant.common.order import Order
from roboquant.common.signal import Signal
from roboquant.traders._util import round_down
from .trader import Trader
from ..common.account import Account
from ..common.event import PriceItem

logger = logging.getLogger(__name__)


class _PositionChange(Flag):
    """representing the four types of changes to the positions in the account.
    This class is used to make the logic in `FlexTrader` easier to understand.
    """

    ENTRY_LONG = auto()
    ENTRY_SHORT = auto()
    EXIT_LONG = auto()
    EXIT_SHORT = auto()

    _ENTRY = ENTRY_LONG | ENTRY_SHORT
    _EXIT = EXIT_LONG | EXIT_SHORT

    @property
    def is_entry(self) -> bool:
        """Return True is the status is open, False otherwise"""
        return self in _PositionChange._ENTRY

    @property
    def is_exit(self) -> bool:
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

    def log_received(self, **kwargs: Any) -> None:
        if logger.isEnabledFor(logging.INFO):
            extra = " ".join(f"{k}={v}" for k, v in kwargs.items())
            logger.info(
                "==> %s received %s",
                self.time,
                extra
            )

    def log_orders(self, orders: list[Order]) -> None:
        """Log an exit due to a signal being converted into an order"""
        if logger.isEnabledFor(logging.INFO):
            logger.info(
                "<== %s converter signal into order(s) %s",
                self.time,
                orders,
            )

    def log_rule(self, rule: str, **kwargs: Any) -> None:
        """Log an exit due to a signal being discarded by a triggered rule"""
        if logger.isEnabledFor(logging.INFO):
            extra = " ".join(f"{k}={v}" for k, v in kwargs.items())
            logger.info(
                "<== %s discarded signal because of '%s' rule %s",
                self.time,
                rule,
                extra,
            )

@dataclass
class FlexTrader(Trader):
    """Implementation of a Trader that has configurable rules to determine which signals are converted into orders.
    This implementation will not generate orders if there is not a price in the event for the underlying asset.

    It does support SignalType.ENTRY, SignalType.EXIT and SignalType.ENTRY_EXIT signals. Also, the signal rating value
    is used to determine the size of the order. A rating of 1.0 means a full BUY order, a rating of 0.5 means half a BUY order
    and a rating of -1.0 means a full SELL order.

    This implementation is designed to be flexible and can be configured to support different trading strategies. The
    default configuration is designed to be safe and conservative, but it can be configured to be more aggressive by changing
    the parameters. The configurable parameters include:

    - one_order_only: don't create new orders for an asset if there is already an open orders for that same asset
    - size_fractions: enable fractional order sizes (if size_fractions is larger than 0), default is 0
    - safety_margin_pct: the safety margin as percentage of equity that should remain available (to avoid margin calls),
    default is 0.05 (5%)
    - max_position_pct: the max percentage of the equity to allocate to a single position, default is 0.1 (10%)
    - max_order_pct: the max percentage of the equity to allocate to a new order, default is 0.05 (5%)
    - min_order_pct: the min percentage of the equity to allocate to a new order, default is 0.02 (2%)
    - shorting: allow orders that could result in a short position, default is false
    - price_type: the price type to use when determining order value, for example "CLOSE". Default is "DEFAULT"
    - shuffle_signals: shuffle the signals before processing them, default is false
    - limit_offset_pct: the offset as percentage for the order limit price. A value of 0.01 means the limit price will be
    1% below market price for buy orders and 1% above the market price for SELL orders. Default is 0.0.
    - tif: the time-in-force policy to use, default is `DAY`

    It might be sometimes challenging to understand why a signal isn't converted into an order. The flex-trader logs
    at INFO level when certain rules have been fired. Enable higher logging:
    ```
        logging.basicConfig()
        logging.getLogger("roboquant.traders.flextrader").setLevel(logging.INFO)
    ```
    """

    one_order_only: bool = True
    size_fractions: int = 0
    safety_margin_pct: float = 0.05
    shorting: bool = False
    max_order_pct: float = 0.05
    min_order_pct: float = 0.02
    max_position_pct: float = 0.1
    price_type: str = "DEFAULT"
    shuffle_signals: bool = False
    limit_offset_pct: float | None = 0.0
    limit_rounding: int = 2
    tif: Literal["DAY", "GTC"] = "DAY"

    def _get_order_size(self, rating: float, contract_price: float, max_order_value: float) -> Decimal:
        """Return the order size"""
        size = Decimal(rating * max_order_value / contract_price)
        return round_down(size, self.size_fractions)

    @override
    def create_orders(self, signals: list[Signal], event: Event, account: Account) -> list[Order]:
        # pylint: disable=too-many-branches,too-many-statements,too-many-locals
        if not signals:
            return []

        if self.shuffle_signals:
            random.shuffle(signals)

        orders: list[Order] = []
        equity = account.equity_value()
        max_order_value = equity * self.max_order_pct
        min_order_value = equity * self.min_order_pct
        max_pos_value = equity * self.max_position_pct
        available = account.buying_power.value - self.safety_margin_pct * equity
        order_assets = {order.asset for order in account.orders}
        ctx = _Context(event.time)

        for signal in signals:
            asset = signal.asset
            pos_size = account.position_size(asset)
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

                rounded_size = round(-pos_size * abs(Decimal(signal.rating)), self.size_fractions)
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

                position_value = account.convert(account.position_amount(asset))
                available_order_value = min(available, max_order_value, max_pos_value - abs(position_value))
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
                    ctx.log_orders(new_orders)
                    orders += new_orders
                    available -= order_value

        return orders


    def _get_limit(self, item: PriceItem, size: Decimal) -> float | None:
        """Calculate the order limit"""
        if self.limit_offset_pct is None:
            return None
        multiplier = 1.0 - self.limit_offset_pct if size > 0 else 1.0 + self.limit_offset_pct
        price = item.price(self.price_type) * multiplier
        limit = round(price, self.limit_rounding)
        return limit

    def _get_orders(self, asset: Asset, size: Decimal, item: PriceItem, signal: Signal, time: datetime) -> list[Order]:
        # pylint: disable=unused-argument
        """Return zero or more orders for the provided asset and size.
        The default implementation:
        - creates a single order
        - with the limit price being the `self.price_type` rounded to two decimals
        - the tif is set to the default "DAY"

        Overwrite this method if you want to implement different logic.
        """
        limit = self._get_limit(item, size)
        result = [Order(asset, size, limit, self.tif)]
        return result

    def __str__(self) -> str:
        attrs = " ".join([f"{k}={v}" for k, v in self.__dict__.items() if not k.startswith("_")])
        return f"FlexTrader({attrs})"
