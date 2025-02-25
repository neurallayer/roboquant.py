import logging
from dataclasses import dataclass

from roboquant.asset import Asset
from roboquant.journals.journal import Journal

logger = logging.getLogger(__name__)


@dataclass
class BasicJournal(Journal):
    """Journal that track the following information:
    - total number of events, and items
    - the total number of signals and orders
    - the maximum open positions
    - the total number of unique assets

    It will also log these values at each step in the run at `info` level.

    This journal adds little overhead to a run, both CPU and memory wise, and is helpful in
    determining if the setup works correctly.
    """

    events: int = 0
    items: int = 0
    orders: int = 0
    signals: int = 0
    assets: int = 0
    max_positions: int = 0

    def __init__(self, log_level=logging.INFO):
        self.__log_level = log_level
        self._assets: set[Asset] = set()

    def track(self, event, account, signals, orders):
        self.items += len(event.items)
        self._assets = self._assets.union(event.price_items.keys())
        self.assets = len(self._assets)
        self.events += 1
        self.signals += len(signals)
        self.orders += len(orders)
        self.max_positions = max(self.max_positions, len(account.positions))

        logger.log(self.__log_level, "time=%s info=%s", event.time, self)
