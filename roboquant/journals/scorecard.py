from collections import defaultdict
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


class ScoreCard(Journal):
    """Tracks progress of a run so it can be plotted using matplotlib charts afterwards.
    It will track the following aspects:
    - the price of a single asset
    - the orders for that asset
    - any metric that is provided
    """

    def __init__(self, *metrics: Metric, include_prices: bool = True, price_type: str = "DEFAULT"):
        super().__init__()
        self._include_prices = include_prices
        self._price_type = price_type
        self._step = 0
        self.metrics = metrics
        self._prices: dict[Asset, _Timeseries] = defaultdict(_Timeseries)
        self._buy_orders: dict[Asset, _Timeseries] = defaultdict(_Timeseries)
        self._sell_orders: dict[Asset, _Timeseries] = defaultdict(_Timeseries)
        self._metric_results: dict[str, _Timeseries] = defaultdict(_Timeseries)

    def track(self, event: Event, account: Account, signals: List[Signal], orders: List[Order]) -> None:
        time = event.time
        if self._include_prices:
            for asset, price in event.get_prices(self._price_type).items():
                self._prices[asset].add(time, price)

            for order in orders:
                if order.is_buy:
                    self._buy_orders[order.asset].add(time, order.limit)
                elif order.is_sell:
                    self._sell_orders[order.asset].add(time, order.limit)

        for metric in self.metrics:
            result = metric.calc(event, account, signals, orders)
            for name, value in result.items():
                self._metric_results[name].add(time, value)

    def plot(self, **kwargs):
        """Plot a chart with the following sub-charts:
        - prices of the configured asset. Orders als small green up (BUY) and red down (SELL) triangles.
        - metrics that have been configured, each in their own chart.
        """
        from matplotlib import pyplot as plt

        ratios = [5 for _ in self._prices] + [2 for _ in self._metric_results]
        fig, axes = plt.subplots(
            len(self._prices) + len(self._metric_results), sharex=True, gridspec_kw={"height_ratios": ratios}
        )

        if not hasattr(axes, "__getitem__"):
            axes = [axes]

        fig.set_size_inches(8.27, 11.69)  # A4
        fig.tight_layout()

        plot_nr = 0

        for asset, timeseries in self._prices.items():
            ax = axes[plot_nr]
            ax.plot(timeseries.time, timeseries.data, **kwargs)  # type: ignore
            ax.set_title(asset.symbol)
            buy_orders = self._buy_orders[asset]
            ax.scatter(buy_orders.time, buy_orders.data, marker="^", color="green")

            sell_orders = self._sell_orders[asset]
            ax.scatter(sell_orders.time, sell_orders.data, marker="v", color="red")

            plot_nr += 1

        for name, value in self._metric_results.items():
            ax = axes[plot_nr]
            ax.tick_params(axis="y", which="major", labelsize="small")
            ax.plot(value.time, value.data)
            ax.set_title(name, y=1.0, pad=-14, size="small")
            plot_nr += 1
