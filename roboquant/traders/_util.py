from roboquant.common.position import Position
from decimal import Decimal
from datetime import datetime
import decimal

from roboquant.common.monetary import Amount
from roboquant.common.signal import Signal


def round_number(value: float | str | decimal.Decimal | int, base: str | Decimal, rounding: str | None = decimal.ROUND_DOWN):
    """
    Flexible rounding of a number to a multiple of the provided base. By default, it will round down (towards zero).
    Internally used for the rounding order sizes and limits.

    For example:
    ```
    round_number(3.1899, "0.01") # 3.18
    round_number(3.1899, "0.05") # 3.15
    ```
    """
    value = Decimal(value)
    base = Decimal(base)
    return base * (value / base).quantize(1, rounding=rounding)


def get_order_size(signal: Signal, price: float, order_amount: Amount, time: datetime, step_size: str) -> decimal.Decimal:
    """Calculate the order size based on the signal rating, asset price, order amount.
    Time is used when a conversion is needed between the asset currency and the order amount currency.
    """
    one_contract_value = signal.asset.amount(decimal.Decimal(1), price).convert_to(order_amount.currency, time)
    size = signal.rating * order_amount.value / one_contract_value
    return round_number(size, step_size)


def is_close(signal: Signal, position: Position):
    """Is the signal opposite to the position"""
    if signal.asset != position.asset:
        return False
    return (position.is_long and signal.is_sell) or (position.is_short and signal.is_buy)


class Sizing:
    """Determine signal impact on the position sizing.
    Takes into account the Signal type (entry/exit), buy or sell and the existing net position size.
    """

    def __init__(self, signal: Signal, pos_size: Decimal):
        self.signal = signal
        self.size = pos_size

    def is_exit(self) -> bool:
        """Return True if this is close of a position, False otherwise"""
        if not self.signal.is_exit or self.size.is_zero():
            return False
        return self.signal.is_buy if self.size < 0 else self.signal.is_sell

    def is_enter(self) -> bool:
        """Return True if this signal would be an opening of a position,
        False otherwise.
        Opening a position can be both Short and Long.
        """
        return self.signal.is_entry and self.size.is_zero()

    def is_increase(self) -> bool:
        """Return True if this an increase into a position size, False otherwise."""
        if not self.signal.is_entry:
            return False
        if self.size.is_zero():
            return True
        return self.signal.is_buy if self.size > 0 else self.signal.is_sell

    def is_shorting(self) -> bool:
        """Return True if this an opening or increasing a short position, False otherwise."""
        return self.signal.is_entry and self.size <= 0 and self.signal.is_sell

    def close_positions(self, positions: list[Position]) -> list[Position]:
        """Get all positions that this signal would exit"""
        result = []
        if not self.signal.is_exit:
            return []
        for pos in positions:
            if is_close(self.signal, pos):
                result.append(pos)
        return result
