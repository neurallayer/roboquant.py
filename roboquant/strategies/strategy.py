from abc import ABC, abstractmethod

from roboquant.event import Event
from roboquant.signal import Signal


class Strategy(ABC):
    """A strategy creates signals based on incoming events and the items these events contain.

    Often these items represent market data, but other types of items are also possible.
    """

    @abstractmethod
    def create_signals(self, event: Event) -> dict[str, Signal]:
        """Create a signal for zero or more symbols. Signals are returned as a dictionary with key being the symbol and
        the value being the Signal.
        """
        ...

    def reset(self):
        """Reset the state of the strategy"""
