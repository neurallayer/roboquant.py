from dataclasses import dataclass
from roboquant.journals.metric import Metric


@dataclass(slots=True)
class RunMetric(Metric):
    """
    Calculates a number of basic metrics of a run:
    - total number of events
    - total number of items
    - total number of signals
    - total number of orders
    """

    events: int = 0  # Total number of events processed
    items: int = 0   # Total number of items processed
    orders: int = 0  # Total number of orders processed
    signals: int = 0 # Total number of signals processed

    def calc(self, event, account, signals, orders) -> dict[str, float]:
        """
        Update the metrics based on the provided event, account, signals, and orders.

        Args:
            event: The event containing items to be processed.
            account: The account information (not used in this method).
            signals: The list of signals generated.
            orders: The list of orders generated.

        Returns:
            A dictionary with the updated metrics.
        """
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
