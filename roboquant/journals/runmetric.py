from dataclasses import dataclass
from roboquant.journals.metric import Metric


@dataclass(slots=True)
class RunMetric(Metric):
    """Calculates a number of basic metrics during a run:
    - total number of events, items, signals and orders
    """

    events: int = 0
    items: int = 0
    orders: int = 0
    signals: int = 0

    def calc(self, event, account, signals, orders) -> dict[str, float]:
        self.items += len(event.items)
        self.events += 1
        self.signals += len(signals)
        self.orders += len(orders)

        return {
            "run/items": self.items,
            "run/signals": self.signals,
            "run/orders": self.orders,
            "run/events": self.events
        }
