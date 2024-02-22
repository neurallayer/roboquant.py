from roboquant.account import Account
from roboquant.event import Event
from roboquant.order import Order
from roboquant.signal import Signal
from roboquant.trackers import Tracker


class TensorboardTracker(Tracker):
    """Logs metrics to a Tensorboard compatible file.
    This can be used outside the realm of machine learning, but requires torch to be installed.
    """

    __slots__ = ("items", "signals", "orders", "writer", "_step")

    def __init__(self, summary_writer):
        self.items: int = 0
        self.signals: int = 0
        self.orders: int = 0
        self.writer = summary_writer
        self._step = 0

    def log(self, event: Event, account: Account, signals: dict[str, Signal], orders: list[Order]):
        self.items += len(event.items)
        self.signals += len(signals)
        self.orders += len(orders)

        self.writer.add_scalar("Run/items", self.items, self._step)
        self.writer.add_scalar("Run/signals", self.signals, self._step)
        self.writer.add_scalar("Run/orders", self.orders, self._step)
        self.writer.add_scalar("Run/equity", account.equity, self._step)
        self._step += 1

    def close(self):
        self.writer.flush()
        self.writer.close()
