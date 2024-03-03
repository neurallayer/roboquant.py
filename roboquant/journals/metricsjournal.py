from datetime import datetime

from roboquant.journals.journal import Journal
from roboquant.journals.metric import Metric
from roboquant.journals.pnlmetric import PNLMetric


class MetricsJournal(Journal):
    """
    A journal that allows for metrics to be added and calculated at each step. It will store
    the results in memory.

    The calculated metric values can be retrieved via the `get_timeseries` method.
    """

    def __init__(self, *metrics: Metric):
        self.metrics = metrics
        self._history: list[tuple[datetime, dict]] = []

    @classmethod
    def pnl(cls):
        return cls(PNLMetric())

    def track(self, event, account, signals, orders):
        result = {}
        for metric in self.metrics:
            new_result = metric.calc(event, account, signals, orders)
            result.update(new_result)

        self._history.append((event.time, result))

    def get_timeseries(self, metric_name: str) -> tuple[list[datetime], list[float]]:
        """Return the calculated values of a metric as tuple of date-times and float values"""
        timeline = []
        values = []
        for time, metrics in self._history:
            if metric_name in metrics:
                timeline.append(time)
                values.append(metrics[metric_name])
        return timeline, values

    def plot(self, plt, metric_name: str, plot_x: bool = True, **kwargs):
        """Plot a metric"""
        x, y = self.get_timeseries(metric_name)
        if plot_x:
            plt.plot(x, y, **kwargs)
        else:
            plt.plot(y, **kwargs)

        if hasattr(plt, "set_title"):
            # assume we are in a subplot
            plt.set_title(metric_name)
        else:
            plt.title(metric_name)

    def get_metric_names(self) -> set[str]:
        """return the available metric names in this journal"""
        result = set()
        for _, m in self._history:
            result.update(m.keys())
        return result
