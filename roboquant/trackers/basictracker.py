from datetime import datetime
import logging
from .tracker import Tracker
from prettytable import PrettyTable

logger = logging.getLogger(__name__)


class BasicTracker(Tracker):
    """Tracks a number of basic metrics:
    - start- and end-time
    - total number of events, items, signals and orders
    - equity

    This tracker adds little overhead to a run, both CPU and memory wise.
    """

    def __init__(self, price_type="DEFAULT"):
        self.start_time = None
        self.end_time = None
        self.items = 0
        self.orders = 0
        self.signals = 0
        self.events = 0
        self.equity = None
        self.buying_power = 0.0

    def log(self, event, account, signals, orders):

        if self.start_time is None:
            self.start_time = event.time

        self.end_time = event.time
        self.items += len(event.items)
        self.orders += len(orders)
        self.events += 1
        self.signals += len(signals)
        self.equity = account.equity
        self.buying_power = account.buying_power

        if logger.isEnabledFor(logging.INFO):
            logger.info(
                "time=%s events=%s items=%s  signals=%s orders=%s",
                self.end_time,
                self.events,
                self.items,
                self.signals,
                self.orders,
            )

    def __repr__(self) -> str:

        def to_timefmt(time: datetime | None):
            return "-" if time is None else time.strftime("%Y-%m-%d %H:%M:%S")

        p = PrettyTable(["metric", "value"], align="r", float_format=".2")

        p.add_row(["start", to_timefmt(self.start_time)])
        p.add_row(["end", to_timefmt(self.end_time)])
        p.add_row(["events", self.events])
        p.add_row(["items", self.items])
        p.add_row(["signals", self.signals])
        p.add_row(["orders", self.orders])

        return p.get_string()
