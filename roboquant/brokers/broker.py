from abc import ABC, abstractmethod
from datetime import datetime, timedelta, timezone

from roboquant.account import Account
from roboquant.event import Event
from roboquant.order import Order


class Broker(ABC):
    """A broker accepts orders and communicates its latest state through the account object when
    the `sync` method is invoked.
    """

    @abstractmethod
    def place_orders(self, orders: list[Order]):
        """
        Place zero or more orders at this broker.

        The following logic applies:

        - If the order doesn't yet have an `id`, it is considered to be a new order and will get assigned a new id.
        - If the order has an `id` and its `size` is zero, it is a cancellation order of an existing order with the same id.
        - If the order has an `id` and its `size` is non-zero, it is an update order of an existing order with the same id.

        Args:
            orders: The orders to be placed.
        """
        ...

    @abstractmethod
    def sync(self, event: Event | None = None) -> Account:
        """Sync the state, and return an updated account to reflect the latest state. So all brokers
        return the same account object, making it easier to switch from back-testing to live-trading.

        Args:
            event: optional the latest event.

        Returns:
            The latest account object.

        """
        ...

    @staticmethod
    def _update_positions(account: Account, event: Event | None, price_type: str = "DEFAULT"):
        """utility methid to update the open positions in the account with the latest market prices found in the event"""
        if not event:
            return

        account.last_update = event.time

        for asset, position in account.positions.items():
            if price := event.get_price(asset, price_type):
                position.mkt_price = price


class LiveBroker(Broker):
    """Base class for brokers that are used in live trading. It contains some common functionality
    that is useful for live brokers.
    """

    def __init__(self) -> None:
        super().__init__()
        self.max_delay = timedelta(minutes=30)

    def guard(self, event: Event | None = None) -> datetime:
        """This method will evaluate an event, and if it occurs to far in the past,
        it will raise a ValueError. Implementations of `LiveBroker` should call this
        method in their `sync` implementation to ensure the `LiveBroker` isn't used
        in a back test.
        """

        now = datetime.now(timezone.utc)

        if not event:
            return now

        if now - event.time > self.max_delay:
            raise ValueError(f"received event too far in the past now={now} event-time={event.time}")

        return now
