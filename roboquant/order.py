from copy import deepcopy
from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Literal

from roboquant.asset import Asset
from roboquant.monetary import Amount


@dataclass(slots=True)
class Order:
    """
    A trading order for an asset. Each order has a mandatory `size` and a `limit` price.
    Orders with a positive `size` are buy orders, and with a negative `size` are sell orders.

    The `gtd` (good till date) is optional, and if not set implies the order is valid
    forever. The `info` can hold any arbitrary properties (kwargs) set on the order.

    The `id` and `fill` are automatically set by the `Broker` and should not be updated.
    """

    asset: Asset
    """The asset of this order."""

    size: Decimal
    """The size (number of contracts) of the order. Positive for buy orders, negative for sell orders.
    """

    limit: float
    """The limit price of the order, denoted in the currency of the asset.
    The limit price is the maximum price you are willing to pay for a buy order,
    or the minimum price you are willing to accept for a sell order.
    Make sure to set the limit price in the currency of the asset and not include more decimal places than
    supported by the broker.
    """

    tif: Literal["GTC", "DAY"]
    """The time in force of the order. `GTC` = Good Till Cancelled, `DAY` = valid for a day only."""

    info: dict[str, Any]
    """Any additional information about the order"""

    id: str
    """The unique id of the order. This is set by the broker only and should not be updated by the user.
    The id is an empty string for new orders and set to a non-empty string when the order is placed with the broker.
    The id is used to identify the order when modifying or cancelling it.
    """

    fill: Decimal
    """The filled size of the order, set by the broker only. Just like the size, positive for buy orders,
    negative for sell orders. So the remaining size is `size - fill`"""

    def __init__(
        self,
        asset: Asset,
        size: Decimal | str | int | float,
        limit: float,
        tif: Literal["GTC", "DAY"] = "DAY",
        **kwargs: Any,
    ):
        """
        Args:
            asset (Asset): The asset of this order.
            size (Decimal | str | int | float): The size of the order. Positive for buy orders, negative for sell orders.
            limit (float): The limit price of the order, denoted in the currency of the asset.
            tif (Literal["GTC", "DAY"], optional): The time in force of the order. Defaults to "DAY".
            **kwargs: Any additional information about the order. It is passed to the broker in the `info` attribute, but not
            maintained over time. Typically used by the broker for additional arguments to their API calls.
        """
        self.asset = asset
        self.size = Decimal(size)
        assert not self.size.is_zero(), "Cannot create a new order with size is zero"

        self.limit = limit
        self.id = ""
        self.fill = Decimal(0)
        self.info = kwargs
        self.tif = tif

    def cancel(self) -> "Order":
        """
        Create a cancellation order. You can only cancel an order that has an `id` assigned to it.
        The returned order is a regular order, but with its `size` set to zero. All additional properties are kept.

        Returns:
            Order: A new order with the same properties but size set to zero.
        """
        assert self.id, "Can only cancel orders with an already assigned id"
        assert self.size, "Cannot cancel a cancellation order, size has to be non-zero"

        result = deepcopy(self)
        result.size = Decimal(0)
        return result

    def modify(self, size: Decimal | str | int | float | None = None, limit: float | None = None) -> "Order":
        """
        Create an modify-order. You can update the size and/or the limit of an order.
        The returned order has the same id as the original order. You can only update existing orders that have an id assigned.

        If you want to cancel an order, use the `cancel` method instead. The size of an order cannot be modified to zero.

        Args:
            size (Decimal | str | int | float | None, optional): The new size of the order.
            limit (float | None, optional): The new limit price of the order.

        Returns:
            Order: A new order with the updated size and limit.
        """
        assert self.id, "Can only update an order with an assigned id"
        assert self.size, "Cannot modify a cancellation order, size has to be non-zero"

        size = Decimal(size) if size is not None else None
        if size is not None:
            assert not size.is_zero(), "size cannot be set to zero, use order.cancel() to cancel an order"

        result = deepcopy(self)
        result.size = size or result.size
        result.limit = limit or result.limit
        return result

    def __deepcopy__(self, _):
        """
        Create a deep copy of the order.

        Args:
            _ : Unused parameter for deepcopy.

        Returns:
            Order: A deep copy of the order.
        """
        result = Order(self.asset, self.size, self.limit, self.tif, **self.info)
        result.id = self.id
        result.fill = self.fill
        return result

    def value(self) -> float:
        """
        Return the total contract value of this order, it ignores the already filled part of the order.

        Returns:
            float: The total contract value of the order.
        """
        return self.asset.contract_value(self.size, self.limit)

    def remaining_value(self) -> float:
        """
        Return the remaining contract value of this order.

        Returns:
            float: The remaining contract value of the order.
        """
        return self.asset.contract_value(self.remaining, self.limit)

    def amount(self) -> Amount:
        """
        Return the total value of this order as a single Amount denoted in the currency of the asset.

        Returns:
            Amount: The total value of the order.
        """
        return Amount(self.asset.currency, self.value())

    @property
    def is_buy(self) -> bool:
        """
        Return True if this is a BUY order, False otherwise.

        Returns:
            bool: True if this is a BUY order, False otherwise.
        """
        return self.size > 0

    @property
    def is_sell(self) -> bool:
        """
        Return True if this is a SELL order, False otherwise.

        Returns:
            bool: True if this is a SELL order, False otherwise.
        """
        return self.size < 0

    @property
    def remaining(self) -> Decimal:
        """
        Return the remaining order size that still needs to be filled.

        Returns:
            Decimal: The remaining order size.
        """
        return self.size - self.fill

    @property
    def is_cancellation(self) -> bool:
        """
        Return True if this is a cancellation order, False otherwise.

        Returns:
            bool: True if this is a cancellation order, False otherwise.
        """
        return self.size.is_zero()

    def is_executable(self, price: float) -> bool:
        """
        Check if this order is executable at the given price.

        Args:
            price (float): The price to check against.

        Returns:
            bool: True if the order is executable at the given price, False otherwise.
        """
        if self.is_buy:
            return price <= self.limit
        return price >= self.limit
