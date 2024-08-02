from abc import ABC, abstractmethod

from roboquant.event import Event
from roboquant.signal import Signal


class Strategy(ABC):
    """A strategy creates orders based on incoming events and the items these events contain.
    Often these items represent market data, but other types of items are also possible.

    Additionally, the account is provided that can help to ensure you don't place orders wihtout
    having the required funding.
    """

    @abstractmethod
    def create_signals(self, event: Event) -> list[Signal]:
        """Create zero or more orders."""
        ...
