from roboquant.journals import Journal
from roboquant.journals.metric import Metric


class TensorboardJournal(Journal):
    """Record metrics to a Tensorboard compatible file.

    This can be used outside the realm of machine learning, but requires tensorboard to be installed.
    """

    def __init__(self, writer, *metrics: Metric):
        """
        Parameters:
            writer: a tensorboard writer instance (`tensorboard.summary.Writer`)
            metrics: the metrics that should be calculated and be added to the tensorboard writer
        """
        self.__writer = writer
        self._step = 0
        self.metrics = metrics

    def track(self, event, account, signals, orders):
        time = event.time.timestamp()
        for metric in self.metrics:
            result = metric.calc(event, account, signals, orders)
            for name, value in result.items():
                self.__writer.add_scalar(name, value, self._step, wall_time=time)

        self._step += 1
