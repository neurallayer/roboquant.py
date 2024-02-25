from typing import Protocol

from roboquant.account import Account
from roboquant.event import Event
from roboquant.order import Order
from roboquant.signal import Signal


class Journal(Protocol):
    """
    A journal allows for the tracking and/or logging of one or more metrics during a run.
    """

    def track(self, event: Event, account: Account, signals: dict[str, Signal], orders: list[Order]):
        """invoked at each step of a run that provides the journal with the opportunity to
        track and log various metrics."""
        ...
