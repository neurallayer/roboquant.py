from typing import Protocol

from roboquant.account import Account
from roboquant.event import Event
from roboquant.order import Order


class Broker(Protocol):
    """A broker handles the placed orders and communicates its state through the account object"""

    def place_orders(self, *orders: Order):
        """
        Place zero or more orders at this broker.

        The following logic applies:
        - if the order doesn't yet have an `id`, it is a new order
        - if the order has an `id` and its `size` is zero, it is a cancellation order
        - if the order has an `id` and its `size` is non-zero, it is an update order
        """
        ...

    def sync(self, event: Event | None = None) -> Account:
        """Sync the state, and return an updated account to reflect the latest state."""
        ...
