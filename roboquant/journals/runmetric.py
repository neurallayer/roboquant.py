from roboquant.journals.metric import Metric


class RunMetric(Metric):
    """Calculates a number of basic metrics during a run:
    - total number of events, items, signals and orders
    """

    def __init__(self):
        super().__init__()
        self.items = 0
        self.orders = 0
        self.signals = 0
        self.events = 0

    def calc(self, event, account, orders) -> dict[str, float]:
        self.items += len(event.items)
        self.orders += len(orders)
        self.events += 1

        return {
            "run/items": self.items,
            "run/orders": self.orders,
            "run/events": self.events,
        }
