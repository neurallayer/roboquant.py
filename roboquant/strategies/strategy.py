from abc import ABC, abstractmethod

from roboquant.event import Event
from roboquant.signal import Signal


class Strategy(ABC):
    """Responsible for creating signals based on incoming events.

    So a strategy doesn't generate the orders, that is the responsibility of
    a `Trader`.

    Often the items in the event represent market data and the strategy uses
    this to perform (technical) analysis. But it is also possible for
    events to contain different data and for example perform fundamental analysis.
    """

    @abstractmethod
    def create_signals(self, event: Event) -> list[Signal]:
        """Create zero or more signals given the provided event.
        args:
            event: The event containing items to be processed.
        returns:
            A list of signals generated based on the event.
            Returns an empty list if no signals are created.
        """
        ...
