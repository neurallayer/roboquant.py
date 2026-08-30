from abc import ABC, abstractmethod

from roboquant.common.account import Account
from roboquant.common.event import Event
from roboquant.common.order import Order


class Broker(ABC):
    """A broker accepts orders and communicates its latest state through returning the `Account` object when
    the `sync` method is invoked.
    """

    @abstractmethod
    def place_orders(self, orders: list[Order]) -> None:
        """
        Place zero or more orders at this broker.

        The following order logic applies:
        - If the order doesn't yet have an `id`, it is considered to be a new order and will get assigned a new id.
        - If the order has an `id` and its `size` is zero, it is a cancellation order of an existing order with the same id.
        - If the order has an `id` and its `size` is non-zero, it is an update order of an existing order with the same id.

        Args:
            orders: The orders to be placed.
        """
        ...

    @abstractmethod
    def sync(self, event: Event | None = None) -> Account:
        """Sync the state and return an updated account to reflect the latest state. So all brokers
        return the same account object, making it easy to switch from back-testing to live-trading.

        Args:
            event: optional the latest event.

        Returns:
            The latest state of the account.
        """
        ...


