import logging
from dataclasses import dataclass

from roboquant.journals.journal import Journal

logger = logging.getLogger(__name__)


@dataclass
class BasicJournal(Journal):
    """Journal that tracks the following metrics:
    - total number of events, and items,
    - the total number of signals and orders
    - the maximum open positions

    It will also log these values at each step in the run at `info` level.

    This journal adds little overhead to a run, both CPU and memory wise, and is helpful in
    determining if the setup works correctly.
    """

    events: int = 0
    items: int = 0
    orders: int = 0
    signals: int = 0
    max_positions: int = 0

    def __init__(self, log_level=logging.INFO):
        self.__log_level = log_level

    def track(self, event, account, signals, orders):
        self.items += len(event.items)
        self.events += 1
        self.signals += len(signals)
        self.orders += len(orders)
        self.max_positions = max(self.max_positions, len(account.positions))

        logger.log(self.__log_level, "time=%s info=%s", event.time, self)
