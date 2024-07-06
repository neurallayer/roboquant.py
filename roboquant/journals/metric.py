from abc import ABC, abstractmethod

from roboquant.account import Account
from roboquant.event import Event
from roboquant.order import Order


class Metric(ABC):
    """Metric calculates zero or more values during each step of a run"""

    @abstractmethod
    def calc(self, event: Event, account: Account, orders: list[Order]) -> dict[str, float]:
        """Calculate zero or more metrics and return the result as a dictionary"""
        ...
