from roboquant.trackers.metric import Metric


class RunMetric(Metric):
    """Tracks a number of basic progress metrics:
    - total number of events, items, signals and orders

    This tracker adds little overhead to a run, both CPU and memory wise.
    """

    def __init__(self):
        self.items = 0
        self.orders = 0
        self.signals = 0
        self.events = 0

    def calc(self, event, account, signals, orders) -> dict[str, float]:
        self.items += len(event.items)
        self.orders += len(orders)
        self.events += 1
        self.signals += len(signals)

        return {
            "run/items": self.items,
            "run/orders": self.orders,
            "run/events": self.events,
            "run/signals": self.signals,
        }
