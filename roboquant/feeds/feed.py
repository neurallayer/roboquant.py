from datetime import datetime
import logging
import threading
from abc import ABC, abstractmethod
from typing import Any

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
        (Re-)plays the events in the feed and puts them on the provided event channel.

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

    def get_ohlcv(self, asset: Asset, timeframe: Timeframe | None = None) -> dict[str, list[float | datetime]]:
        """Get the OHLCV values for an asset in this feed.
        The returned value is a dict with the keys being "Date", "Open", "High", "Low", "Close", "Volume"
        and the values a list.
        """

        result = {column: [] for column in ["Date", "Open", "High", "Low", "Close", "Volume"]}
        channel = self.play_background(timeframe)
        while event := channel.get():
            item = event.price_items.get(asset)
            if item and isinstance(item, Bar):
                result["Date"].append(event.time)
                result["Open"].append(item.ohlcv[0])
                result["High"].append(item.ohlcv[1])
                result["Low"].append(item.ohlcv[2])
                result["Close"].append(item.ohlcv[3])
                result["Volume"].append(item.ohlcv[4])
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
        """Return the prices of one or more assets as a dict with the key being the synbol name."""

        assert assets, "provide at least 1 asset"
        result = {asset.symbol: [] for asset in assets}
        channel = self.play_background(timeframe)
        while evt := channel.get():
            for asset in assets:
                price = evt.get_price(asset, price_type)
                result[asset.symbol].append(price)
        return result

    def plot(self, asset: Asset, price_type: str = "DEFAULT", timeframe: Timeframe | None = None, plt: Any = None, **kwargs):
        """
        Plot the prices of a single asset. This requires matplotlib to be installed.

        Parameters
        ----------
        asset : Asset
            The asset for which to plot prices.
        price_type : str, optional
            The type of price to plot, e.g. open, close, high, low. (default is "DEFAULT")
        timeframe : Timeframe or None, optional
            The timeframe over which to plot prices. If None, the entire feed timeframe is used. (default is None)
        plt : matplotlib axes
            The matplotlib.pyplot object where the plot will be drawn. If none is specified, the default pyplot will be used
        **kwargs
            Additional keyword arguments to pass to the plt.plot() function.
        """
        if not plt:
            from matplotlib import pyplot as plt

        times, prices = self.get_asset_prices(asset, price_type, timeframe)
        plt.plot(times, prices, **kwargs)

        if hasattr(plt, "set_title"):
            plt.set_title(asset.symbol)
        elif hasattr(plt, "title"):
            plt.title(asset.symbol)

        return plt

    def get_asset_prices(
        self, asset: Asset, price_type="DEFAULT", timeframe: Timeframe | None = None
    ) -> tuple[list[datetime], list[float]]:
        """Get prices for a single asset from the feed by replaying the feed."""

        x = []
        y = []
        channel = self.play_background(timeframe)
        while event := channel.get():
            price = event.get_price(asset, price_type)
            if price:
                x.append(event.time)
                y.append(price)
        return x, y
