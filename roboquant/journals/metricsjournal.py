from typing import Self, override
from datetime import datetime

from roboquant.common.metric import Metric
from roboquant.journals.journal import Journal
from roboquant.util.metrics import PNLMetric
from roboquant.common.account import Account
from roboquant.common.signal import Signal
from roboquant.common.event import Event
from roboquant.common.order import Order
from typing import Dict, List

from roboquant.common.timeseries import TimeSeries

class MetricsJournal(Journal):
    """
    Implementation of a journal that allows for metrics to be added and captured at each step. It will store
    the results of the metrics in memory.

    The calculated metric values can be retrieved via `get_metric`. There is also
    convenience method to plot a metric.
    """

    def __init__(self, *metrics: Metric) -> None:
        self.metrics = metrics
        self._history: list[tuple[datetime, dict[str, float]]] = []

    @classmethod
    def pnl(cls) -> Self:
        """Return a metrics journal pre-configured with the PNL metric"""
        return cls(PNLMetric())

    @override
    def track(self, event: Event, account: Account, signals: List[Signal], orders: List[Order]) -> None:
        result: Dict[str, float] = {}
        for metric in self.metrics:
            new_result = metric.calc(event, account, signals, orders)
            result.update(new_result)

        if result:
            self._history.append((event.time, result))

    def get_metrics(self, *metric_names: str) -> TimeSeries:
        """Return the ccaptured metrics of oen or more metrics as a TimeSeries"""
        timeline: list[datetime] = []
        values = {name: [] for name in metric_names}
        for time, metrics in self._history:
            for name in metric_names:
                value = metrics.get(name, float("nan"))
                values[name].append(value)
            timeline.append(time)

        return TimeSeries.from_data(timeline, values)

    def get_metric_names(self) -> list[str]:
        """Return a list of the recorded metric names"""
        result: set[str] = set()
        for _, m in self._history:
            result.update(m.keys())
        return list(result)

    def plot(self, metric_name: str, plot_timeline: bool = True, ax = None, **kwargs):
        """Plot the metric. Optional a `matplotlib.axes.Axes` can be provided
        This method requires matplotlib to be installed."""

        ts = self.get_metrics(metric_name)
        return ts.plot(plot_timeline=plot_timeline,ax=ax, **kwargs)


