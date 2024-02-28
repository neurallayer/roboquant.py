from typing import Protocol

from roboquant.account import Account
from roboquant.event import Event
from roboquant.order import Order


class Broker(Protocol):
    """A broker accepts orders and communicates its state through the account object"""

    def place_orders(self, orders: list[Order]):
        """
        Place zero or more orders at this broker.

        The following logic applies:

        - If the order doesn't yet have an `id`, it is considered to be a new order and will get assigned a new id.
        - If the order has an `id` and its `size` is zero, it is a cancellation order of an existing order with the same id.
        - If the order has an `id` and its `size` is non-zero, it is an update order of an existing order with the same id.

        Args:
            orders: The orders to be placed.
        """
        ...

    def sync(self, event: Event | None = None) -> Account:
        """Sync the state, and return an updated account to reflect the latest state.

        Args:
            event: optional the latest event.

        Returns:
            The latest account object.

        """
        ...
