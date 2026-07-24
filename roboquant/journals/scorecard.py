from roboquant.feeds.historic import InMemoryFeed
from roboquant.journals.journal import Journal
from roboquant.journals.metric import Metric
from roboquant.event import Event, TradePrice
from roboquant.account import Account
from roboquant.journals.metricsjournal import MetricsJournal
from roboquant.signal import Signal
from roboquant.order import Order
from typing import Any, List


class ScoreCard(Journal):
    """Tracks progress of a run so it can be plotted using matplotlib charts afterwards.
    It will track the following aspects:
    - the price of the assets
    - the orders for that asset as markers on the price chart
    - any additional metric that has been provided

    This works best on smaller runs with a limited number of assets and orders.
    """

    def __init__(self, *metrics: Metric, include_prices: bool = True, price_type: str = "DEFAULT") -> None:
        super().__init__()
        self._include_prices = include_prices
        self._price_type = price_type
        self._step = 0
        self.metrics = metrics
        self._feed = InMemoryFeed()
        self._trades = []
        self._journal = MetricsJournal(*metrics)

    def track(self, event: Event, account: Account, signals: List[Signal], orders: List[Order]) -> None:
        if self._include_prices:
            for asset, price in event.get_prices(self._price_type).items():
                self._feed._add_item(event.time, TradePrice(asset, price))

        self._trades = account.trades
        self._journal.track(event, account, signals, orders)

    def plot(self, size: tuple[float, float] = (8.27, 11.69), **kwargs: Any) -> None:
        """Plot a chart with the following sub-charts:
        - prices of the configured asset. Orders als small green up (BUY) and red down (SELL) triangles.
        - metrics that have been configured, each in their own chart.
        """
        from matplotlib import pyplot as plt
        self._feed._update()

        ratios = [5 for _ in self._feed.assets()] + [2 for _ in self._journal.get_metric_names()]
        fig, axes = plt.subplots(
            len(self._feed.assets()) + len(self._journal.get_metric_names()),
            sharex=True, gridspec_kw={"height_ratios": ratios}
        )

        if not hasattr(axes, "__getitem__"):
            axes = [axes]

        fig.set_size_inches(size)
        fig.tight_layout()

        plot_nr = 0

        if self._include_prices:
            for asset in self._feed.assets():
                ax = axes[plot_nr]
                self._feed.plot(asset, ax=ax, trades = self._trades)
                plot_nr += 1

        for name in self._journal.get_metric_names():
            ax = axes[plot_nr]
            self._journal.plot(name, ax=ax)
            plot_nr += 1
