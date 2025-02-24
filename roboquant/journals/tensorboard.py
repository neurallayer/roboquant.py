from roboquant.journals.journal import Journal
from roboquant.journals.metric import Metric


class TensorboardJournal(Journal):
    """Record metrics to a Tensorboard compatible file. The wall time is set to the event time, so
    with the right configuration in the tensorboard UI, you can see the metrics evolve over time.

    This can be used outside the realm of machine learning, but requires tensorboard library to be installed.
    """

    def __init__(self, writer, *metrics: Metric):
        """
        Parameters:
            writer: a tensorboard writer instance (`tensorboard.summary.Writer`)
            metrics: the metrics that should be calculated and be added to the tensorboard writer
        """
        super().__init__()
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
