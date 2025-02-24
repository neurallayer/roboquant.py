from abc import ABC, abstractmethod

from roboquant.account import Account
from roboquant.event import Event
from roboquant.order import Order
from roboquant.signal import Signal


class Journal(ABC):
    """
    A journal enables the tracking and/or logging of progress during a run.

    A journal can hold detailed records of all your trading activities in the financial markets.
    It serves as a tool to track the performance, decisions, and outcomes over the timeline
    of a run.
    """

    @abstractmethod
    def track(self, event: Event, account: Account, signals: list[Signal], orders: list[Order]):
        """This method is invoked at each step of a run and provides the journal with the opportunity to
        track and log various metrics."""
        ...
