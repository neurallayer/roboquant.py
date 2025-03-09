from datetime import datetime
from matplotlib import pyplot as plt
from roboquant.asset import Asset
from roboquant.journals.journal import Journal
from roboquant.journals.metric import Metric


class _DataHolder:

    def __init__(self):
        self.x = []
        self.y = []

    def add(self, time: datetime, data):
        self.x.append(time)
        self.y.append(data)

class PlotJournal(Journal):
    """Tracks progress of a run so it can be plotted using matplotlib afterwards."""

    def __init__(self, asset: Asset, *metrics: Metric):
        super().__init__()
        self._asset = asset
        self._step = 0
        self.metrics = metrics
        self._prices = _DataHolder()
        self._buy_signals = []
        self._sell_signals = []
        self._buy_orders = _DataHolder()
        self._sell_orders = _DataHolder()
        self._metrics = {}


    def track(self, event, account, signals, orders):
        time = event.time
        if price := event.get_price(self._asset):
            self._prices.add(time, price)

        for signal in signals:
            if signal.asset == self._asset:
                if signal.is_buy:
                    self._buy_signals.append(time)
                else:
                    self._sell_signals.append(time)

        for order in orders:
            if order.asset == self._asset:
                if order.is_buy:
                    self._buy_orders.add(time, order.limit)
                else:
                    self._sell_orders.add(time, order.limit)

        for metric in self.metrics:
            result = metric.calc(event, account, signals, orders)
            for name, value in result.items():
                if name not in self._metrics:
                    self._metrics[name] = _DataHolder()
                self._metrics[name].add(time, value)

    def plot(self, **kwargs):
        """Plot a chart with the following charts:
        - prices of the configured asset.
        - signals als horizontal lines, greeen and red.
        - orders als small green up and red down triangles.
        - metrics that have been configured.

        """
        ratios = [5,] + [1 for _ in self._metrics]
        fig, axes = plt.subplots(1 + len(self._metrics), sharex=True, gridspec_kw={'height_ratios': ratios})
        # fig.subplots_adjust(hspace=0)
        # fig.set_size_inches(15, 5 + (2 * len(ratios)))
        fig.set_size_inches(8.27, 11.69) # A4
        fig.tight_layout()

        plot_nr = 0
        ax = axes[plot_nr]
        result = ax.plot(self._prices.x, self._prices.y, **kwargs)  # type: ignore
        ax.set_title(self._asset.symbol)

        for signal in self._buy_signals:
            ax.axvline(x = signal, color="green")

        for signal in self._sell_signals:
            ax.axvline(x = signal, color="red")

        ax.scatter(self._buy_orders.x, self._buy_orders.y, marker="^", color = "green")
        ax.scatter(self._sell_orders.x, self._sell_orders.y, marker="v", color = "red")

        for name, value in self._metrics.items():
            plot_nr+=1
            ax = axes[plot_nr]
            ax.tick_params(axis='y', which='major', labelsize="small")
            ax.plot(value.x, value.y)
            ax.set_title(name, y=1.0, pad=-14, size="small")

        return result
