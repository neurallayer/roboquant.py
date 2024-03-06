import threading
from abc import ABC, abstractmethod

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
            try:
                self.play(channel)
            except ChannelClosed:
                # this exception we can expect
                pass
            finally:
                channel.close()

        thread = threading.Thread(None, __background, daemon=True)
        thread.start()
        return channel

    def plot(self, plt, symbol: str, price_type: str = "DEFAULT", timeframe: Timeframe | None = None, **kwargs):
        """
        Plot the prices of a symbol.

        Parameters
        ----------
        plt : matplotlib.pyplot
            The matplotlib.pyplot object where the plot will be drawn.
        symbol : str
            The symbol for which to plot prices.
        price_type : str, optional
            The type of price to plot, e.g. open, close, high, low. (default is "DEFAULT")
        timeframe : Timeframe or None, optional
            The timeframe over which to plot prices. If None, the entire feed timeframe is used. (default is None)
        **kwargs
            Additional keyword arguments to pass to the plt.plot() function.
        """

        channel = self.play_background(timeframe)
        times = []
        prices = []
        while evt := channel.get():
            price = evt.get_price(symbol, price_type)
            if price is not None:
                times.append(evt.time)
                prices.append(price)

        plt.plot(times, prices, **kwargs)
        if hasattr(plt, "set_title"):
            # assume we are in a subplot
            plt.set_title(symbol)
        else:
            plt.title(symbol)
