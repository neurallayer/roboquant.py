from dataclasses import dataclass
from roboquant.journals.metric import Metric
from typing import List, Dict
from roboquant.event import Event
from roboquant.account import Account
from roboquant.signal import Signal
from roboquant.order import Order


@dataclass(slots=True)
class RunMetric(Metric):
    """
    Calculates a number of basic metrics of a run:
    - total number of events in the run
    - total number of items in the events
    - total number of signals generated
    - total number of orders created
    """

    events: int = 0  # Total number of events processed
    items: int = 0   # Total number of items processed
    orders: int = 0  # Total number of orders processed
    signals: int = 0 # Total number of signals processed

    def calc(self, event: Event, account: Account, signals: List[Signal], orders: List[Order]) -> Dict[str, float]:
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
