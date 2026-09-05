from abc import ABC, abstractmethod

from roboquant.common.account import Account
from roboquant.common.event import Event
from roboquant.common.order import Order


class Broker(ABC):
    """A broker accepts orders and reports its current state via the `Account` object returned from the
    the `sync` method is invoked.

    A broker is responsible for the actual execution of orders and for keeping track of the
    current state of the account, including cash balances, positions, and open orders.

    Typical use cases are:
    - Paper trading or back-testing, where a simulated broker fills orders based on market data.
    - Live trading, where a broker forwards orders to an external exchange or broker/dealer API.

    Implementations should be careful to preserve the exact order semantics defined in
    `place_orders`, since strategies rely on those semantics to open, update, and cancel orders.
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
