from copy import copy
from dataclasses import dataclass
from decimal import Decimal
from typing import Any


@dataclass(slots=True)
class Order:
    """
    A trading order.
    The `id` is automatically assigned by the broker and should not be set manually.
    Also, the `status` and `fill` are managed by the broker and should not be manually set.
    """

    symbol: str
    size: Decimal
    limit: float
    info: dict[str, Any]

    id: str | None = None
    fill: Decimal = Decimal(0)

    def __init__(
        self, symbol: str, size: Decimal | str | int | float, limit: float, **kwargs
    ):
        self.symbol = symbol
        self.size = Decimal(size)
        assert not self.size.is_zero(), "Cannot create a new order with size is zero"

        self.limit = limit
        self.id: str | None = None
        self.fill = Decimal(0)
        self.info = kwargs

    def required_funding(self, position: Decimal = Decimal()) -> float:
        if abs(self.remaining + position) < abs(position):
            return 0.0
        return abs(self.limit * float(self.remaining))

    def cancel(self) -> "Order":
        """Create a cancellation order. You can only cancel orders that are still open and have an id.
        The returned order looks like a regular order, but with its size set to zero.
        """
        assert self.id is not None, "Can only cancel orders with an already assigned id"
        result = copy(self)
        result.size = Decimal(0)
        return result

    def modify(self, size: Decimal | str | int | float | None = None, limit: float | None = None) -> "Order":
        """Create an update-order. You can update the size and/or limit of an order. The returned order has the same id
        as the original order.

        You can only update existing orders that are still open and have an id.
        """

        assert self.id, "Can only update an already assigned id"
        size = Decimal(size) if size is not None else None
        if size is not None:
            assert not size.is_zero(), "size cannot be set to zero, use order.cancel() to cancel an order"

        result = copy(self)
        result.size = size or result.size
        result.limit = limit or result.limit
        return result

    def __copy__(self):
        result = Order(self.symbol, self.size, self.limit, **self.info)
        result.id = self.id
        result.fill = self.fill
        return result

    @property
    def is_cancellation(self):
        """Return True if this is a cancellation order, False otherwise"""
        return self.size.is_zero()

    @property
    def is_buy(self):
        """Return True if this is a BUY order, False otherwise"""
        return self.size > 0

    @property
    def is_sell(self):
        """Return True if this is a SELL order, False otherwise"""
        return self.size < 0

    @property
    def remaining(self):
        """Return the remaining order size to be filled.

        In case of a sell order, the remaining can be a negative number.
        """
        return self.size - self.fill
