from datetime import datetime
from roboquant.asset import Asset
from roboquant.journals.journal import Journal
from roboquant.journals.metric import Metric
from roboquant.event import Event
from roboquant.account import Account
from roboquant.signal import Signal
from roboquant.order import Order
from typing import List


class _Timeseries:

    def __init__(self):
        self.time: list[datetime] = []
        self.data: list[float] = []

    def add(self, time: datetime, data: float):
        self.time.append(time)
        self.data.append(data)

class ChartingJournal(Journal):
    """Tracks progress of a run so it can be plotted using matplotlib charts afterwards.
    It will track the following aspects:
    - the price of a single asset
    - the orders for that asset
    - any metric that is provided
    """

    def __init__(self, asset: Asset, *metrics: Metric):
        super().__init__()
        self._asset = asset
        self._step = 0
        self.metrics = metrics
        self._prices = _Timeseries()
        self._buy_orders = _Timeseries()
        self._sell_orders = _Timeseries()
        self._metric_results: dict[str, _Timeseries] = {}


    def track(self, event: Event, account: Account, signals: List[Signal], orders: List[Order]) -> None:
        time = event.time
        if price := event.get_price(self._asset):
            self._prices.add(time, price)

        for order in orders:
            if order.asset == self._asset:
                if order.is_buy:
                    self._buy_orders.add(time, order.limit)
                elif order.is_sell:
                    self._sell_orders.add(time, order.limit)

        for metric in self.metrics:
            result = metric.calc(event, account, signals, orders)
            for name, value in result.items():
                if name not in self._metric_results:
                    self._metric_results[name] = _Timeseries()
                self._metric_results[name].add(time, value)

    def plot(self, **kwargs):
        """Plot a chart with the following sub-charts:
        - prices of the configured asset. Orders als small green up (BUY) and red down (SELL) triangles.
        - metrics that have been configured, each in their own chart.
        """
        from matplotlib import pyplot as plt
        ratios = [5,] + [1 for _ in self._metric_results]
        fig, axes = plt.subplots(1 + len(self._metric_results), sharex=True, gridspec_kw={'height_ratios': ratios})

        if not hasattr(axes, "__getitem__"):
            axes = [axes]

        # fig.subplots_adjust(hspace=0)
        # fig.set_size_inches(15, 5 + (2 * len(ratios)))
        fig.set_size_inches(8.27, 11.69) # A4
        fig.tight_layout()

        plot_nr = 0
        ax = axes[plot_nr]
        result = ax.plot(self._prices.time, self._prices.data, **kwargs)  # type: ignore
        ax.set_title(self._asset.symbol)

        ax.scatter(self._buy_orders.time, self._buy_orders.data, marker="^", color = "green")
        ax.scatter(self._sell_orders.time, self._sell_orders.data, marker="v", color = "red")

        for name, value in self._metric_results.items():
            plot_nr+=1
            ax = axes[plot_nr]
            ax.tick_params(axis='y', which='major', labelsize="small")
            ax.plot(value.time, value.data)
            ax.set_title(name, y=1.0, pad=-14, size="small")

        return result
