from abc import ABC, abstractmethod
from datetime import datetime
from itertools import chain
from typing import Any, Generator, Sequence, override

from roboquant.asset import Asset
from roboquant.event import Bar, Event, PriceItem
from roboquant.timeframe import Timeframe
from roboquant.timeseries import TimeSeries
from .feed import Feed


class HistoricFeed(Feed, ABC):
    """Base class for most implementations of Historic Feeds. Contains several methods to enhance feeds,
    like plotting prices and conversion to dataframes."""

    @abstractmethod
    def assets(self) -> list[Asset]:
        ...

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
            ValueError: If no asset is found with the specified symbol.
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
        This is mostly useful for debugging purposes to see what items a feed generates.
        """

        for event in self.play(timeframe):
            print(event.time)
            for item in event.items:
                print("======> ", item)

    def count_events(self, timeframe: Timeframe | None = None, include_empty: bool=False) -> int:
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

    def to_dict(
        self, *assets: Asset, timeframe: Timeframe | None = None, price_type: str = "DEFAULT"
    ) -> dict[str, list[float | None]]:
        """Return the prices of one or more assets as a dict with the key being the symbol name."""

        assert assets, "provide at least 1 asset"
        result: dict[str, list[float | None]] = {asset.symbol: [] for asset in assets}
        for evt in self.play(timeframe):
            for asset in assets:
                price = evt.get_price(asset, price_type)
                result[asset.symbol].append(price)
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
        timeframe: Timeframe | None = None,
        ax = None,
        **kwargs,
    ):
        """
        Plots the prices of a single asset. This function requires matplotlib to be installed.

        Args:
            asset (Asset | str): The asset or symbol for which to plot prices.
            price_type (str, optional): The type of price to plot, e.g., "OPEN" or "CLOSE". Defaults to "DEFAULT".
            timeframe (Timeframe | None, optional): The timeframe over which to plot prices. If None, the entire feed
                timeframe is used. Defaults to None.
            ax (matplotlib.axes.Axes, optional): The matplotlib axis where the plot will be drawn. If not specified,
                the default pyplot axis will be used.
            **kwargs: Additional keyword arguments to pass to the `ax.plot()` function.

        Returns:
            list: The result of the `ax.plot()` function, which is a list of Line2D objects.
        """

        ts = self.get_prices(asset, price_type, timeframe)
        result = ts.plot(ax=ax, **kwargs)
        return result


    def get_prices(self, asset: Asset | str, price_type : str ="DEFAULT", timeframe: Timeframe | None = None
    ) -> TimeSeries:
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
            Timeseries with the name being the symbol name of the asset.
        """
        x :list[datetime] = []
        y : list[float] = []

        if isinstance(asset, str):
            asset = self.get_asset(asset)

        for event in self.play(timeframe):
            price = event.get_price(asset, price_type)
            if price:
                x.append(event.time)
                y.append(price)
        return TimeSeries(asset.symbol, x, y)

    def __repr__(self) -> str:
        feed = self.__class__.__name__
        return f"{feed}(assets={len(self.assets())} timeframe={self.timeframe()})"



class InMemoryFeed(HistoricFeed, ABC):
    """
    Base class for feeds that contain historic market data and store them in-memory.
    Internally, it uses a sorted-by-datetime dictionary to store the market data.
    """

    def __init__(self):
        super().__init__()
        self.__data: dict[datetime, list[PriceItem]] = {}
        self.__modified: bool = False
        self.__assets: set[Asset] = set()

    def _add_item(self, dt: datetime, item: PriceItem):
        """Add a price-item at a moment in time to this feed.
        Subclasses should invoke this method to populate the historic-feed.

        Items added at the same time will be part of the same event.
        So each unique time will only produce a single event.
        """
        self.__modified = True

        if dt not in self.__data:
            self.__data[dt] = [item]
        else:
            items = self.__data[dt]
            items.append(item)

    @override
    def assets(self) -> list[Asset]:
        """Return the list of unique assets available in this feed"""
        return list(self.__assets)

    def timeline(self) -> list[datetime]:
        """Return the timeline of this feed as a list of datatime objects"""
        return list(self.__data.keys())

    def timeframe(self) -> Timeframe:
        """Return the timeframe of this feed"""
        tl = self.timeline()
        if tl:
            return Timeframe(tl[0], tl[-1], inclusive=True)

        return Timeframe.EMPTY

    def _update(self):
        """invoke this method once all historic data has been added, so internal state
        can be updated.
        """

        if self.__modified:
            self.__data = dict(sorted(self.__data.items()))
            price_items = chain.from_iterable(self.__data.values())
            self.__assets = {item.asset for item in price_items}
            self.__modified = False

    def get_first_event(self) -> Event | None:
        """Return the first event in this feed, or None if no events are available"""
        if not self.__data:
            return None

        first_time = next(iter(self.__data.keys()))
        items = self.__data[first_time]
        return Event(first_time, items)

    def get_last_event(self) -> Event | None:
        """Return the last event in this feed, or None if no events are available"""
        if not self.__data:
            return None

        last_time = next(reversed(self.__data.keys()))
        items = self.__data[last_time]
        return Event(last_time, items)

    def play(self, timeframe: Timeframe | None = None) -> Generator[Event, Any, None]:
        for k, v in self.__data.items():
            if not timeframe or k in timeframe:
                yield Event(k, v)
            elif k <= timeframe.start:
                continue
            else:
                break
