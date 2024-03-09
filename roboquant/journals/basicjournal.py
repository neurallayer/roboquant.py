import logging
from dataclasses import dataclass

from roboquant.journals.journal import Journal

logger = logging.getLogger(__name__)


@dataclass
class BasicJournal(Journal):
    """Tracks a number of basic metrics:
    - total number of events, items, signals and orders until that time

    It will also log these values at each step in the run at `info` level.

    This journal adds little overhead to a run, both CPU and memory wise.
    """

    items: int
    orders: int
    signals: int
    events: int

    def __init__(self):
        self.events = 0
        self.signals = 0
        self.items = 0
        self.orders = 0

    def track(self, event, account, signals, orders):
        self.items += len(event.items)
        self.events += 1
        self.signals += len(signals)
        self.orders += len(orders)

        logger.info("time=%s info=%s", event.time, self)
