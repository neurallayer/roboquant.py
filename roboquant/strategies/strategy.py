from abc import ABC, abstractmethod

from roboquant.account import Account
from roboquant.event import Event
from roboquant.order import Order


class Strategy(ABC):
    """A strategy creates signals based on incoming events and the items these events contain.

    Often these items represent market data, but other types of items are also possible.
    """

    @abstractmethod
    def create_orders(self, event: Event, account: Account) -> list[Order]:
        """Create a signal for zero or more symbols. Signals are returned as a dictionary with key being the symbol and
        the value being the Signal.
        """
        ...

    def reset(self):
        """Reset the state of the strategy"""
