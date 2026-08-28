from abc import ABC
from datetime import datetime
from typing import Any, Iterable

from matplotlib.axes import Axes

from roboquant.common.trade import Trade
from roboquant.common.asset import Asset
from roboquant.common.event import Bar
from roboquant.common.timeframe import Timeframe
from roboquant.common.timeseries import TimeSeries
from roboquant.common.metric import Metric
from .feed import Feed

import matplotlib.pyplot as plt


class HistoricFeed(Feed, ABC):
    """Base class for most implementations of Historic Feeds. Contains several methods
    to enhance feeds, like plotting prices and conversion to dataframes."""

    def symbols(self) -> list[str]:
        """Return the list of unique symbols available in this feed"""
        symbols = set()
        for asset in self.assets():
            symbols.add(asset.symbol)
        return list(symbols)

    def get_asset(self, symbol: str) -> Asset:
        """Retrieve the first asset that matches the provided symbol name.

        Args:
            symbol (str): The symbol name of the asset to retrieve.
        Returns:
            Asset: The first asset object that matches the provided symbol.
        Raises:
            ValueError: If no asset is found matching the symbol.
        """
        try:
            return next(asset for asset in self.assets() if asset.symbol == symbol)
        except StopIteration:
            raise ValueError(f"no asset found with symbol={symbol}")

    def get_ohlcv(self, asset: Asset | str, timeframe: Timeframe | None = None) -> TimeSeries:
        """Get the OHLCV values for a single asset in this feed.
        The returned value is a TimeSeries.
        """

        if isinstance(asset, str):
            asset = self.get_asset(asset)

        timeline = []
        data: dict[str, list[float]] = {}
        keys = ["open", "high", "low", "close", "volume"]
        for key in keys:
            data[key] = []

        for event in self.play(timeframe):
            item = event.price_items.get(asset)
            if item and isinstance(item, Bar):
                timeline.append(event.time)
                for idx, key in enumerate(keys):
                    data[key].append(item.ohlcv[idx])

        return TimeSeries.from_data(timeline, data)

    def print_items(self, timeframe: Timeframe | None = None) -> None:
        """Print the items in a feed to the console.
        This is mostly useful for debugging purposes to
        see what items a feed generates.
        """

        for event in self.play(timeframe):
            print(event.time)
            for item in event.items:
                print("======> ", item)

    def count_events(self, timeframe: Timeframe | None = None, include_empty: bool = False) -> int:
        """Count the number of events in a feed"""

        events = 0
        for evt in self.play(timeframe):
            if evt.items or include_empty:
                events += 1
        return events

    def count_items(self, timeframe: Timeframe | None = None) -> int:
        """Count the number of events in a feed"""

        items = 0
        for evt in self.play(timeframe):
            items += len(evt.items)
        return items

    def to_timeseries(
            self, *assets: Asset, timeframe: Timeframe | None = None, price_type: str = "DEFAULT"
        ) -> TimeSeries:
            """Return the prices of one or more assets as a multivariate TimeSeries.
            The name of each individual series is the symbol name.
            If at a moment in time for an asset there is no known price, NaN will be stored.

            If no assets are provided, all assets in the feed will be used.
            """
            if not assets:
                assets = tuple(self.assets())

            timeline = []
            result: dict[str, list[float]] = {asset.symbol: [] for asset in assets}
            for evt in self.play(timeframe):
                timeline.append(evt.time)
                for asset in assets:
                    price = evt.get_price(asset, price_type)
                    if price is not None:
                        result[asset.symbol].append(price)
                    else:
                        result[asset.symbol].append(float("nan"))
            return TimeSeries.from_data(timeline, result)

    def plot(
        self,
        asset: Asset | str,
        price_type: str = "DEFAULT",
        volume_type: str = "DEFAULT",
        timeframe: Timeframe | None = None,
        ax: Axes | None = None,
        trades: Iterable[Trade] | None = None,
        plot_volume: bool = True,
        **kwargs: Any,
    ) -> Axes:
        """
        Plots the prices of a single asset. This function requires matplotlib to be installed.
        It also supports plotting trades on the same chart,

        Args:
            asset (Asset | str): The asset or symbol for which to plot prices.
            price_type (str, optional): The type of price to plot, e.g., "OPEN" or "CLOSE". Defaults to "DEFAULT".
            timeframe (Timeframe | None, optional): The timeframe over which to plot prices. If None, the entire feed
                timeframe is used. Defaults to None.
            ax (matplotlib.axes.Axes, optional): The matplotlib axis where the plot will be drawn. If not specified,
                the default pyplot axis will be used.
            trades: trades to be plotted as markers.
            **kwargs: Additional keyword arguments to pass to the `ax.plot()` function.

        Returns:
            list: The result of the `ax.plot()` function, which is a list of Line2D objects.
        """
        if isinstance(asset, str):
            asset = self.get_asset(asset)

        t: list[datetime] = []
        p: list[float] = []
        v: list[float] = []

        for event in self.play(timeframe):
            if item := event.price_items.get(asset):
                t.append(event.time)
                p.append(item.price(price_type))
                if plot_volume:
                    v.append(item.volume(volume_type))

        if not ax:
            _, ax = plt.subplots()
            ax.set_title(asset.symbol)


        ax.plot(t, p, **kwargs)  # type: ignore

        if plot_volume:
            ax2 = ax.twinx()
            ax2.grid(False)
            ax2.bar(t, v, alpha=0.3)  # type: ignore

        if trades and t:
            tf = Timeframe(t[0], t[-1], True)
            trades = [t for t in trades if t.asset == asset and t.time in tf]

            buy = [t for t in trades if t.size > 0]
            if buy:
                x = [t.time for t in buy]
                y = [t.price for t in buy]
                ax.scatter(x, y, marker="^", color="limegreen", zorder=10)  # type: ignore

            sell = [t for t in trades if t.size < 0]
            if sell:
                x = [t.time for t in sell]
                y = [t.price for t in sell]
                ax.scatter(x, y, marker="v", color="red", zorder=10)  # type: ignore

        return ax

    def track(self, metric: Metric, timeframe: Timeframe | None = None) -> TimeSeries:
        """
        Track a metric over time and return the results as a TimeSeries.
        The metric will only be provided the event and empty values for the other
        parameters in tis `calc()` method.
        """
        from roboquant.common.account import Account

        timeline: list[datetime] = []
        account = Account.empty()
        data: dict[str, list[float]] = {}

        for event in self.play(timeframe):
            result = metric.calc(event, account, [], [])
            if result:
                for key, value in result.items():
                    if key not in data:
                        data[key] = []
                    data[key].append(value)
                timeline.append(event.time)

        return TimeSeries.from_data(timeline, data)
