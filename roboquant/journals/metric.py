from abc import ABC, abstractmethod

from roboquant.account import Account
from roboquant.event import Event
from roboquant.order import Order
from roboquant.signal import Signal


class Metric(ABC):
    """Metric calculates zero or more values during each step of a run.
    They can be used for example in the MetricsJournal.
    """

    @abstractmethod
    def calc(self, event: Event, account: Account, signals: list[Signal], orders: list[Order]) -> dict[str, float]:
        """Calculate zero or more metrics and return the result as a dictionary. The dictionary should not be modified
        after it is returned. The keys in the dictionary should be unique and not conflict with other metrics."""
        ...
