from dataclasses import dataclass
from datetime import datetime
import logging

from roboquant.trackers.tracker import Tracker
from roboquant.account import Account
from roboquant.event import Event
from roboquant.order import Order
from roboquant.signal import Signal

logger = logging.getLogger(__name__)


@dataclass
class BasicTracker(Tracker):
    """Tracks a number of basic metrics:
    - last time
    - total number of events, items, signals and orders until that time

    This tracker adds little overhead to a run, both CPU and memory wise.
    """
    time: datetime | None
    items: int
    orders: int
    signals: int
    events: int

    def __init__(self, output=False):
        self.time = None
        self.items = 0
        self.orders = 0
        self.signals = 0
        self.events = 0
        self.__output = output

    def trace(self, event: Event, account: Account, signals: dict[str, Signal], orders: list[Order]):
        self.time = event.time
        self.items += len(event.items)
        self.orders += len(orders)
        self.events += 1
        self.signals += len(signals)

        if self.__output:
            print(self.__repr__() + "\n")
