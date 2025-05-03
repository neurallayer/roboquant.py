from abc import ABC, abstractmethod
from datetime import timedelta
import time

from roboquant.account import Account
from roboquant.event import Event
from roboquant.order import Order
from roboquant.timeframe import Timeframe, utcnow


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
        """Sync the state and return an updated account to reflect the latest state. So all brokers
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
    """
    Base class for brokers used in live trading.
    This class provides common functionality for live brokers, such as throttling,
    safety checks, and metrics tracking. It ensures that live trading operations
    are performed with extra precautions and includes methods for synchronizing
    account state and managing orders.

    Attributes:
        max_delay_event (timedelta): Maximum allowable delay for processing an event.
        max_delay_sync (timedelta): Maximum allowable delay for synchronizing account state.
        sleep_after_cancel (float): Time to sleep (in seconds) after canceling an order.
        metrics (dict): A dictionary tracking the number of new, updated, canceled,
            and synchronized orders.
    Methods:
        sync(event: Event | None = None) -> Account:
            Synchronizes the account state with the broker, ensuring it is up to date.
        _get_account() -> Account:
            Abstract method to retrieve the current account state. Must be implemented
            by subclasses.
        _cancel_order(order: Order):
            Abstract method to cancel an order. Must be implemented by subclasses.
        _update_order(order: Order):
            Abstract method to update an order. Must be implemented by subclasses.
        _place_order(order: Order):
            Abstract method to place a new order. Must be implemented by subclasses.
        place_orders(orders: list[Order]):
            Processes a list of orders, handling new, updated, and canceled orders.
    """

    def __init__(self) -> None:
        super().__init__()
        self.max_delay_event = timedelta(minutes=20)
        self.max_delay_sync = timedelta(seconds=5)
        self.sleep_after_cancel = 0.0

        self._has_new_orders = False
        self._account: Account = Account()
        self._account.last_update = Timeframe.INFINITE.start
        self.metrics = {
            "new": 0,
            "update": 0,
            "cancel": 0,
            "sync": 0,
        }

    def sync(self, event: Event | None = None) -> Account:
        now = utcnow()

        # Safety check for not using this real broker in a back test over historic data
        if event and (now - event.time > self.max_delay_event):
            raise ValueError(f"received event too far in the past now={now} event-time={event.time}")

        diff_time = now - self._account.last_update

        # We always get a new account state if we just placed orders
        # or if we are due to refresh
        if self._has_new_orders or diff_time > self.max_delay_sync:
            self.metrics["sync"] += 1
            self._account = self._get_account()
            self._account.last_update = now
            self._has_new_orders = False

        return self._account

    @abstractmethod
    def _get_account(self) -> Account:
        """subclasses should implement this method"""
        pass

    @abstractmethod
    def _cancel_order(self, order: Order):
        """subclasses should implement this method"""
        pass

    @abstractmethod
    def _update_order(self, order: Order):
        """subclasses should implement this method"""
        pass

    @abstractmethod
    def _place_order(self, order: Order):
        """subclasses should implement this method"""
        pass

    def place_orders(self, orders: list[Order]):
        if not orders:
            return

        self._has_new_orders = True
        for order in orders:
            if order.id and order.size.is_zero():
                self.metrics["cancel"] += 1
                self._cancel_order(order)
                time.sleep(self.sleep_after_cancel)
            elif order.id:
                self.metrics["update"] += 1
                self._update_order(order)
            elif not order.id and not order.size.is_zero():
                self.metrics["new"] += 1
                self._place_order(order)
            else:
                raise ValueError(f"Invalid order {order}")
