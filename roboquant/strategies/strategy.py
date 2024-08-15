from abc import ABC, abstractmethod

from roboquant.event import Event
from roboquant.signal import Signal


class Strategy(ABC):
    """A strategy creates signals based on incoming events and the items within these events.
    Often the items represent market data associated with an asset, but other types of items
    are also possible.
    """

    @abstractmethod
    def create_signals(self, event: Event) -> list[Signal]:
        """Create zero or more signals for provided event."""
        ...
