from abc import ABC, abstractmethod
from datetime import datetime, timedelta, timezone

from roboquant.account import Account
from roboquant.event import Event
from roboquant.order import Order


class Broker(ABC):
    """A broker accepts orders and communicates its state through the account object"""

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
        """Sync the state, and return an updated account to reflect the latest state.

        Args:
            event: optional the latest event.

        Returns:
            The latest account object.

        """
        ...


class BaseBroker(Broker):
    """A broker accepts orders and communicates its state through the account object"""

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
        """Sync the state, and return an updated account to reflect the latest state.

        Args:
            event: optional the latest event.

        Returns:
            The latest account object.

        """
        ...


def _update_account(account: Account, event: Event | None, price_type: str = "DEFAULT"):

    if not event:
        return

    account.last_update = event.time

    for symbol, position in account.positions.items():
        if price := event.get_price(symbol, price_type):
            position.mkt_price = price


def _update_positions(account: Account, event: Event | None, price_type: str = "DEFAULT"):
    """update the open positions in the account with the latest market prices"""
    if not event:
        return

    account.last_update = event.time

    for symbol, position in account.positions.items():
        if price := event.get_price(symbol, price_type):
            position.mkt_price = price


class LiveBroker(Broker):

    def __init__(self) -> None:
        super().__init__()
        self.max_delay = timedelta(minutes=30)

    def guard(self, event: Event | None = None) -> datetime:

        now = datetime.now(timezone.utc)

        if not event:
            return now

        if now - event.time > self.max_delay:
            raise ValueError(f"received event too far in the past now={now} event-time={event.time}")

        return now
