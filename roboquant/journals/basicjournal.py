import logging
from dataclasses import dataclass

from roboquant.journals.journal import Journal

logger = logging.getLogger(__name__)


@dataclass
class BasicJournal(Journal):
    """Tracks a number of basic metrics:
    - total number of events, items, signalsm, orders and max positions until that time

    It will also log these values at each step in the run at `info` level.

    This journal adds little overhead to a run, both CPU and memory wise and is helfull in
    determning if the setup works correctly.
    """

    items: int
    orders: int
    signals: int
    events: int
    max_positions: int

    def __init__(self):
        self.events = 0
        self.signals = 0
        self.items = 0
        self.orders = 0
        self.max_positions = 0

    def track(self, event, account, signals, orders):
        self.items += len(event.items)
        self.events += 1
        self.signals += len(signals)
        self.orders += len(orders)
        self.max_positions = max(self.max_positions, len(account.positions))

        logger.info("time=%s info=%s", event.time, self)
