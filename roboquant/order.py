from copy import copy
from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any

from roboquant.asset import Asset
from roboquant.wallet import Amount


@dataclass(slots=True)
class Order:
    """
    A trading order.
    The `id` is automatically assigned by the broker and should not be set manually.
    Also, the `fill` are managed by the broker and should not be manually set.
    """

    asset: Asset
    size: Decimal
    limit: float
    info: dict[str, Any]

    id: str | None
    fill: Decimal
    created_at: datetime | None

    def __init__(self, asset: Asset, size: Decimal | str | int | float, limit: float, **kwargs):
        self.asset = asset
        self.size = Decimal(size)
        assert not self.size.is_zero(), "Cannot create a new order with size is zero"

        self.limit = limit
        self.id: str | None = None
        self.fill = Decimal(0)
        self.info = kwargs
        self.created_at = None

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
        as the original order. You can only update existing orders that have an id.
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
        result = Order(self.asset, self.size, self.limit, **self.info)
        result.id = self.id
        result.fill = self.fill
        result.created_at = self.created_at
        return result

    def value(self):
        return self.asset.contract_value(self.size, self.limit)

    def amount(self):
        return Amount(self.asset.currency, self.value())

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


class OrderUtil:
    """Set of utils for dealing with orders"""

    @staticmethod
    def cancel_old_orders(orders: list[Order], now: datetime, older_than: timedelta):
        result = [order.cancel() for order in orders if order.created_at + older_than < now]
        return result

    @staticmethod
    def find_orders(orders: list[Order], *symbols: str):
        result = [order for order in orders if order.asset.symbol in symbols]
        return result
