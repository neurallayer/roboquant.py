from roboquant.feeds.historic import InMemoryFeed
from roboquant.journals.journal import Journal
from roboquant.journals.metric import Metric
from roboquant.event import Event, TradePrice
from roboquant.account import Account
from roboquant.journals.metricsjournal import MetricsJournal
from roboquant.signal import Signal
from roboquant.order import Order
from typing import List


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

    def plot(self) -> None:
        """Plot 2 charts:
        - prices of the assets in the feed. Trades are small green up (BUY) and red down (SELL) triangle markers.
        - metrics that have been configured and captured.
        """
        from matplotlib import pyplot as plt
        self._feed._update()

        if self._include_prices:
            assets = self._feed.assets()
            rows = len(assets) // 2 + len(assets) % 2

            _ , axs = plt.subplots(rows, 2, figsize=(20, 3 * len(assets)))

            for ax, asset in zip(axs.flatten(), assets):
                ax.grid(True, color="grey", linestyle="--")
                self._feed.plot(asset, ax = ax, trades = self._trades, linewidth=1)


        metric_names = self._journal.get_metric_names()
        rows = len(metric_names) // 2 + len(metric_names) % 2
        _ , axs = plt.subplots(rows, 2, figsize=(20, 3 * len(metric_names)))

        for ax, name in zip(axs.flatten(), metric_names):
            ax.grid(True, color="grey", linestyle="--")
            self._journal.plot(name, ax=ax, linewidth=1)
