from roboquant.trackers import Tracker
from roboquant.trackers.metric import Metric


class TensorboardTracker(Tracker):
    """Record metrics to a Tensorboard compatible file.
    This can be used outside the realm of machine learning, but requires torch to be installed.
    """

    def __init__(self, summary_writer, *metrics: Metric):
        self.writer = summary_writer
        self._step = 0
        self.metrics = metrics

    def track(self, event, account, signals, orders):
        time = event.time.timestamp()
        for metric in self.metrics:
            result = metric.calc(event, account, signals, orders)
            for name, value in result.items():
                self.writer.add_scalar(name, value, self._step, time)

        self._step += 1

    def close(self):
        self.writer.flush()
        self.writer.close()
