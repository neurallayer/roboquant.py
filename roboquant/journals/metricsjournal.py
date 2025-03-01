from datetime import datetime
import matplotlib.axes
from matplotlib import pyplot as plt

from roboquant.journals.journal import Journal
from roboquant.journals.metric import Metric
from roboquant.journals.pnlmetric import PNLMetric



class MetricsJournal(Journal):
    """
    A journal that allows for metrics to be added and calculated at each step. It will store
    the results in memory.

    The calculated metric values can be retrieved via the `get_metric` method. There is also
    functionality to plot a metric.
    """

    def __init__(self, *metrics: Metric):
        self.metrics = metrics
        self._history: list[tuple[datetime, dict[str, float]]] = []

    @classmethod
    def pnl(cls):
        """Return a metrics journal pre-configured with the PNL metric"""
        return cls(PNLMetric())

    def track(self, event, account, signals, orders):
        result = {}
        for metric in self.metrics:
            new_result = metric.calc(event, account, signals, orders)
            result.update(new_result)

        self._history.append((event.time, result))

    def get_metric(self, metric_name: str) -> tuple[list[datetime], list[float]]:
        """Return the calculated values of a metric as tuple of date-times and float values"""
        timeline: list[datetime] = []
        values: list[float] = []
        for time, metrics in self._history:
            if metric_name in metrics:
                timeline.append(time)
                values.append(metrics[metric_name])
        return timeline, values

    def plot(self, metric_name: str, plot_x: bool = True, ax: matplotlib.axes.Axes | None  = None, **kwargs):
        """Plot one of the metrics. Optional a `matplotlib.axes.Axes` can be provided
        This requires matplotlib to be installed."""
        if not ax:
            _, ax = plt.subplots()

        x, y = self.get_metric(metric_name)

        if plot_x:
            result = ax.plot(x, y, **kwargs)  # type: ignore
        else:
            result = ax.plot(y, **kwargs)

        ax.set_title(metric_name)
        return result

    def get_metric_names(self) -> set[str]:
        """return all the recorded metric names"""
        result: set[str] = set()
        for _, m in self._history:
            result.update(m.keys())
        return result
