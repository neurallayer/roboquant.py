from datetime import datetime
import logging
import threading
from abc import ABC, abstractmethod
from typing import Sequence
from matplotlib import pyplot as plt
import matplotlib.axes

from roboquant.asset import Asset
from roboquant.event import Bar
from roboquant.feeds.eventchannel import EventChannel, ChannelClosed
from roboquant.timeframe import Timeframe


class Feed(ABC):
    """
    A Feed represents a source of (financial) events that can be (re-)played.
    It provides methods for playing events, plotting prices, and playing events in the background on a separate thread.
    """

    @abstractmethod
    def play(self, channel: EventChannel):
        """
        (Re-)play the events in the feed and put them on the provided event channel.

        Parameters
        ----------
        channel : EventChannel
            EventChannel where the events will be placed.
        """
        ...

    def timeframe(self) -> Timeframe:
        """Return the timeframe of this feed, default is Timeframe.INFINITE"""
        return Timeframe.INFINITE

    def play_background(self, timeframe: Timeframe | None = None, channel_capacity: int = 10) -> EventChannel:
        """
        Plays this feed in the background on its own thread.

        Parameters
        ----------
        timeframe : Timeframe or None, optional
            The timeframe in which to limit the events in the feed. If None, all events will be played.
        channel_capacity : int, optional
            The capacity of the event channel (default is 10)

        Returns
        -------
        EventChannel
            The EventChannel used for playback.
        """

        channel = EventChannel(timeframe, channel_capacity)

        def __background():
            # pylint: disable=broad-exception-caught
            try:
                self.play(channel)
            except ChannelClosed:
                # this exception we can expect
                pass
            except Exception as e:
                logging.error("Error during playback", exc_info=e)
            finally:
                channel.close()

        thread = threading.Thread(None, __background, daemon=True)
        thread.start()
        return channel

    def get_ohlcv(self, asset: Asset, timeframe: Timeframe | None = None) -> dict[datetime, Sequence[float]]:
        """Get the OHLCV values for an asset in this feed.
        The returned value is a `dict` with the keys being the `datetime` and the value being an `array`
        of the OHLCV values.
        """

        result: dict[datetime, Sequence[float]] = {}
        channel = self.play_background(timeframe)
        while event := channel.get():
            item = event.price_items.get(asset)
            if item and isinstance(item, Bar):
                result[event.time] = item.ohlcv

        return result

    def print_items(self, timeframe: Timeframe | None = None, timeout: float | None = None) -> None:
        """Print the items in a feed to the console.
        This is mostly useful for debugging purposes to see what items a feed generates.
        """

        channel = self.play_background(timeframe)
        while event := channel.get(timeout):
            print(event.time)
            for item in event.items:
                print("======> ", item)

    def count_events(self, timeframe: Timeframe | None = None, timeout: float | None = None, include_empty=False) -> int:
        """Count the number of events in a feed"""

        channel = self.play_background(timeframe)
        events = 0
        while evt := channel.get(timeout):
            if evt.items or include_empty:
                events += 1
        return events

    def count_items(self, timeframe: Timeframe | None = None, timeout: float | None = None) -> int:
        """Count the number of events in a feed"""

        channel = self.play_background(timeframe)
        items = 0
        while evt := channel.get(timeout):
            items += len(evt.items)
        return items

    def to_dict(
        self, *assets: Asset, timeframe: Timeframe | None = None, price_type: str = "DEFAULT"
    ) -> dict[str, list[float | None]]:
        """Return the prices of one or more assets as a dict with the key being the symbol name."""

        assert assets, "provide at least 1 asset"
        result = {asset.symbol: [] for asset in assets}
        channel = self.play_background(timeframe)
        while evt := channel.get():
            for asset in assets:
                price = evt.get_price(asset, price_type)
                result[asset.symbol].append(price)
        return result

    def to_dataframe(self, asset: Asset, timeframe: Timeframe | None = None):
        """Return the bars for the asset as a Pandas dataframe, with the index being the event time
        and the columns being "Open", "High", "Low", "Close", "Volume".

        This will throw an exception if the Pandas library isn't installed.
        """
        import pandas as pd

        ohlcv = self.get_ohlcv(asset, timeframe)
        columns = ["Open", "High", "Low", "Close", "Volume"]
        return pd.DataFrame.from_dict(ohlcv, orient="index", columns=columns)  # type: ignore

    def plot(
        self,
        asset: Asset,
        price_type: str = "DEFAULT",
        timeframe: Timeframe | None = None,
        ax: matplotlib.axes.Axes | None = None,
        **kwargs,
    ):
        """
        Plot the prices of a single asset. This requires matplotlib to be installed. It returns the plotted chart.

        Parameters
        ----------
        asset : Asset
            The asset for which to plot prices.
        price_type : str, optional
            The type of price to plot, e.g. open, close, high, low. (default is "DEFAULT")
        timeframe : Timeframe or None, optional
            The timeframe over which to plot prices. If None, the entire feed timeframe is used. (default is None)
        ax : matplotlib axes
            The matplotlib.pyplot axis where the plot will be drawn. If none is specified, the default pyplot will be used
        **kwargs
            Additional keyword arguments to pass to the plt.plot() function.
        """
        if not ax:
            _, ax = plt.subplots()

        times, prices = self.get_asset_prices(asset, price_type, timeframe)
        result = ax.plot(times, prices, **kwargs)  # type: ignore
        ax.set_title(asset.symbol)
        return result


    def get_asset_prices(self, asset: Asset, price_type="DEFAULT", timeframe: Timeframe | None = None
    ) -> tuple[list[datetime], list[float]]:
        """
        Retrieve the prices of a given asset, optional over a specified timeframe.

        Args:
            asset (Asset): The asset for which to retrieve prices.
            price_type (str, optional): The type of price to retrieve (e.g., "DEFAULT", "CLOSE", "OPEN").
            Defaults to "DEFAULT".
            timeframe (Timeframe | None, optional): The timeframe over which to retrieve prices.
            If None, the entire available timeframe is used. Defaults to None.

        Returns:
            tuple[list[datetime], list[float]]: A tuple containing two lists:
                - A list of datetime objects representing the times at which prices were recorded.
                - A list of float values representing the prices of the asset at the corresponding times.
        """
        x = []
        y = []
        channel = self.play_background(timeframe)
        while event := channel.get():
            price = event.get_price(asset, price_type)
            if price:
                x.append(event.time)
                y.append(price)
        return x, y
