from roboquant.journals.journal import Journal
from roboquant.journals.metric import Metric


class TensorboardJournal(Journal):
    """Record metrics to a Tensorboard compatible file. The wall time is set to the event time, so
    with the right configuration in the tensorboard UI, you can see the metrics evolve over the historic timeline.

    This can be used outside the realm of machine learning, but requires tensorboard library to be installed.
    """

    def __init__(self, writer, *metrics: Metric):
        """
        Initialize the TensorboardJournal.

        Parameters:
            writer: A tensorboard writer instance (`tensorboard.summary.Writer`).
            metrics: The metrics that should be calculated and added to the tensorboard writer.
        """
        super().__init__()
        self.__writer = writer
        self._step = 0
        self.metrics = metrics

    def track(self, event, account, signals, orders):
        """
        Track the metrics and record them to the tensorboard writer.

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
