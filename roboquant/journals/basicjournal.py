import logging
from dataclasses import dataclass

from roboquant.journals.journal import Journal

logger = logging.getLogger(__name__)


@dataclass
class BasicJournal(Journal):
    """Tracks a number of basic metrics:
    - total number of events, items, signals and orders until that time
    - total pnl percentage

    This journal adds little overhead to a run, both CPU and memory wise.
    """
    items: int
    orders: int
    signals: int
    events: int
    pnl: float

    def __init__(self):
        self.items = 0
        self.orders = 0
        self.signals = 0
        self.events = 0
        self.pnl = 0.0
        self.__first_equity = None

    def track(self, event, account, signals, orders):
        if self.__first_equity is None:
            self.__first_equity = account.equity

        self.items += len(event.items)
        self.orders += len(orders)
        self.events += 1
        self.signals += len(signals)
        self.pnl = account.equity / self.__first_equity - 1.0

        logger.info("time=%s info=%s", event.time, self)
