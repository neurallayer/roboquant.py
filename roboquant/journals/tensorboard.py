from typing import Protocol, override

from roboquant.common.account import Account
from roboquant.common.event import Event
from roboquant.journals.journal import Journal
from roboquant.common.metric import Metric
from roboquant.common.order import Order
from roboquant.common.signal import Signal


class Writer(Protocol):

    def add_scalar(self, tag: str, data: float, step: int, *, wall_time: float, description: str | None = None) -> None:
        ...


class TensorboardJournal(Journal):
    """Record metrics to a Tensorboard compatible file.

    Overall it is similar to the `MetricsJournal`, but rather than keeping
    it in memory for futher inspection, it is saved to a file. You can use
    Tensorboard to view this file while the `run` is executing.

    The wall time is set to the event time, so with the right configuration
    in the tensorboard UI, you can see the metrics evolve over the correct historic timeline.

    This can be used outside the realm of machine learning, but requires
    the tensorboard library to be installed.

    Example
    ```
    from tensorboard.summary import Writer
    writer = Writer("./runs")
    journal = TensorboardJournal(writer, RunMetric(), PNLMetric())
    ```
    """

    def __init__(self, writer: Writer, *metrics: Metric) -> None:
        """
        Initialize the TensorboardJournal.

        Args:
            writer: A tensorboard summary writer instance (`tensorboard.summary.Writer` or
            `torch.utils.tensorboard.SummaryWriter`).
            metrics: Metrics that should be calculated at each step and added to the tensorboard writer.
        """
        super().__init__()
        assert hasattr(writer, "add_scalar") and callable(writer.add_scalar), "writer not a tensorboard summary writer"
        self.__writer = writer
        self._step : int = 0
        self.metrics = metrics

    @override
    def track(self, event: Event, account: Account, signals: list[Signal], orders: list[Order]) -> None:
        """
        Calculate the metrics and add them to the tensorboard writer.

        Parameters:
            event: The event containing the time and other relevant information.
            account: The account information.
            signals: The signals generated during the event.
            orders: The orders generated during the event.

        The wall time is set to the event time, and the metrics are recorded with the current step.
        """
        time = event.time.timestamp()
        for metric in self.metrics:
            result = metric.calc(event, account, signals, orders)
            for name, value in result.items():
                self.__writer.add_scalar(name, value, self._step, wall_time=time)

        self._step += 1


