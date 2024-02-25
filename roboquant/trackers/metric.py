from roboquant.account import Account
from roboquant.event import Event
from roboquant.order import Order
from roboquant.signal import Signal


from typing import Protocol


class Metric(Protocol):
    """Metric calculates zero or more values during each step of a run"""

    def calc(self, event: Event, account: Account, signals: dict[str, Signal], orders: list[Order]) -> dict[str, float]:
        """Calculate zero or more metrics and return the result as a dict."""
        ...
