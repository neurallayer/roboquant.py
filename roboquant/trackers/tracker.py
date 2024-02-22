from typing import Protocol
from roboquant.account import Account

from roboquant.event import Event
from roboquant.order import Order
from roboquant.signal import Signal


class Tracker(Protocol):
    """
    A tracker allows for tracking and/or logging of one or more metrics during a run.
    """

    def log(self, event: Event, account: Account, signals: dict[str, Signal], orders: list[Order]):
        """invoked at each step of a run that provides the tracker with the opportunity to
        calculate metrics and log these."""
        ...
