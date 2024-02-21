from typing import Protocol

from roboquant.event import Event
from roboquant.signal import Signal


class Strategy(Protocol):
    """A strategy creates signals based on incoming events and the items these events contain."""

    def create_signals(self, event: Event) -> dict[str, Signal]:
        """Create a signal for zero or more symbols. Signals are returned as a dictionary with key being the symbol name and
        the value being the Signal.
        """
        ...
