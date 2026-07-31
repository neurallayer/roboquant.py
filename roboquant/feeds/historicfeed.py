from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Iterable, Sequence

from roboquant.common.trade import Trade
from roboquant.common.asset import Asset
from roboquant.common.event import Bar
from roboquant.common.timeframe import Timeframe
from roboquant.common.timeseries import TimeSeries
from .feed import Feed


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


class HistoricFeed(Feed, ABC):
    """Base class for most implementations of Historic Feeds. Contains several methods
    to enhance feeds, like plotting prices and conversion to dataframes."""

    @abstractmethod
    def assets(self) -> list[Asset]: ...

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

    def get_ohlcv(self, asset: Asset | str, timeframe: Timeframe | None = None) -> dict[datetime, Sequence[float]]:
        """Get the OHLCV values for an asset in this feed.
        The returned value is a `dict` with the key being the `datetime` and the value being an `array`
        of the OHLCV values.
        """

        if isinstance(asset, str):
            asset = self.get_asset(asset)

        result: dict[datetime, Sequence[float]] = {}
        for event in self.play(timeframe):
            item = event.price_items.get(asset)
            if item and isinstance(item, Bar):
                result[event.time] = item.ohlcv

        return result

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
            return TimeSeries(timeline, result)

    def to_dict(
        self, *assets: Asset, timeframe: Timeframe | None = None, price_type: str = "DEFAULT"
    ) -> dict[str, list[float]]:
        """Return the prices of one or more assets as a dict with the key being the symbol name.
        If at a moment in time for an asset there is no known price, NaN will be stored.

        If no assets are provided, all assets in the feed will be used.
        """
        if not assets:
            assets = tuple(self.assets())

        result: dict[str, list[float]] = {asset.symbol: [] for asset in assets}
        for evt in self.play(timeframe):
            for asset in assets:
                price = evt.get_price(asset, price_type)
                if price is not None:
                    result[asset.symbol].append(price)
                else:
                    result[asset.symbol].append(float("nan"))
        return result

    def to_dataframe(self, asset: Asset | str, timeframe: Timeframe | None = None):
        """Return the bars for the asset as a Pandas dataframe, with the index being the event time
        and the columns being "Open", "High", "Low", "Close", "Volume".

        This will throw an exception if the Pandas library isn't installed.
        """
        import pandas as pd

        ohlcv = self.get_ohlcv(asset, timeframe)
        columns = ["open", "high", "low", "close", "volume"]
        return pd.DataFrame.from_dict(ohlcv, orient="index", columns=columns)  # type: ignore

    def plot(
        self,
        asset: Asset | str,
        price_type: str = "DEFAULT",
        volume_type: str = "DEFAULT",
        timeframe: Timeframe | None = None,
        ax=None,
        trades: Iterable[Trade] | None = None,
        plot_volume: bool = True,
        **kwargs: Any,
    ):
        """
        Plots the prices of a single asset. This function requires matplotlib to be installed.
        It also support plotting trades on the same chart,

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
            ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
            ax.set_title(asset.symbol)

        if not kwargs:
            kwargs = {"linewidth": 1}

        ax.plot(t, p, **kwargs)  # type: ignore

        if plot_volume:
            ax2 = ax.twinx()
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

    def get_prices(self, asset: Asset | str, price_type: str = "DEFAULT", timeframe: Timeframe | None = None) -> TimeSeries:
        """
        Retrieve the prices for a given asset, optional over a specified timeframe and return the result
        as a `TimeSeries`.

        Args:
            asset (Asset): The asset for which to retrieve prices.
            price_type (str, optional): The type of price to retrieve (e.g., "DEFAULT", "CLOSE", "OPEN").
            Defaults to "DEFAULT".
            timeframe (Timeframe | None, optional): The timeframe over which to retrieve prices.
            If None, the entire available timeframe is used. Defaults to None.

        Returns:
            TimeSeries with the name being the symbol name of the asset.
        """
        x: list[datetime] = []
        y: list[float] = []

        if isinstance(asset, str):
            asset = self.get_asset(asset)

        for event in self.play(timeframe):
            price = event.get_price(asset, price_type)
            if price:
                x.append(event.time)
                y.append(price)
        return TimeSeries.univariate(asset.symbol, x, y)

    def plot_corr(
        self, *assets: Asset, ax=None, timeframe: Timeframe | None = None, price_type: str = "DEFAULT",
        fontsize : int | None = None
    ):
        """Plot the correlation matrix of various assets in a feed. If no assets are provided,
        all feed assets are used. Returns the main ax.
        """

        if not ax:
            _, ax = plt.subplots()

        d = self.to_dict(*assets, timeframe=timeframe, price_type=price_type)
        df = pd.DataFrame.from_dict(d)
        corr = df.corr()

        c_axes = ax.matshow(corr, vmin=-1, vmax=1, cmap="RdYlGn")
        ax.figure.colorbar(c_axes)

        columns = corr.columns
        plt.xticks(range(len(columns)), columns, fontsize = fontsize)  # type: ignore
        plt.yticks(range(len(columns)), columns, fontsize = fontsize)  # type: ignore

        for (i, j), z in np.ndenumerate(corr.to_numpy()):
            ax.text(
                j,
                i,
                "{:0.2f}".format(z),
                ha="center",
                va="center",
                color="w",
                fontsize = fontsize,
                bbox=dict(boxstyle="round", facecolor='#222', edgecolor='#333', alpha=0.2),
            )

        return ax
