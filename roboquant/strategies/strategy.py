from abc import ABC, abstractmethod

from roboquant.account import Account
from roboquant.event import Event
from roboquant.order import Order


class Strategy(ABC):
    """A strategy creates orders based on incoming events and the items these events contain.
    Often these items represent market data, but other types of items are also possible.

    Additionally, the account is provided that can help to ensure you don't place orders wihtout
    having the required funding.
    """

    @abstractmethod
    def create_orders(self, event: Event, account: Account) -> list[Order]:
        """Create zero or more orders."""
        ...
