from datetime import datetime
from typing import Any

from roboquant.journals.journal import Journal
from roboquant.journals.metric import Metric
from roboquant.journals.pnlmetric import PNLMetric

try:
    from matplotlib import pyplot
except ImportError:
    pyplot = None


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
        """Return a metrics journal configured with the PNL metric"""
        return cls(PNLMetric())

    def track(self, event, account, signals, orders):
        result = {}
        for metric in self.metrics:
            new_result = metric.calc(event, account, signals, orders)
            result.update(new_result)

        self._history.append((event.time, result))

    def get_metric(self, metric_name: str) -> tuple[list[datetime], list[float]]:
        """Return the calculated values of a metric as tuple of date-times and float values"""
        timeline = []
        values = []
        for time, metrics in self._history:
            if metric_name in metrics:
                timeline.append(time)
                values.append(metrics[metric_name])
        return timeline, values

    def plot(self, metric_name: str, plot_x: bool = True, plt: Any = pyplot, **kwargs):
        """Plot a metric"""
        assert plt, "no plt explicitly specified or matplotlib found"

        x, y = self.get_metric(metric_name)

        if plot_x:
            plt.plot(x, y, **kwargs)
        else:
            plt.plot(y, **kwargs)

        if hasattr(plt, "set_title"):
            plt.set_title(metric_name)
        elif hasattr(plt, "title"):
            plt.title(metric_name)

        return plt

    def get_metric_names(self) -> set[str]:
        """return the recorded metric names"""
        result = set()
        for _, m in self._history:
            result.update(m.keys())
        return result
