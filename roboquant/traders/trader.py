from abc import ABC, abstractmethod

from roboquant.account import Account
from roboquant.event import Event
from roboquant.order import Order
from roboquant.signal import Signal


class Trader(ABC):
    """A Trader creates orders, typically based on the signals it receives from a strategy.
    But it is also possible to implement all order logic in a Trader and don't rely on signals at all.

    In contrast to a `Strategy`, a `Trader` has also acces the `Account` object and can make
    decisions based on that state. So, for example, it is possible to create sell orders
    for open positions that have a large unrealized loss.
    """

    @abstractmethod
    def create_orders(self, signals: list[Signal], event: Event, account: Account) -> list[Order]:
        """Create zero or more orders.

        Arguments
            signals: Zero or more signals created by the strategy.
            event: The event with its items.
            account: The latest account object.

        Returns:
            A list containing zero or more orders.
        """
        ...
