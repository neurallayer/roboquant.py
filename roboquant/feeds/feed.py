import logging
from datetime import datetime
import threading
from abc import ABC, abstractmethod

from roboquant.feeds.eventchannel import EventChannel, ChannelClosed
from roboquant.timeframe import Timeframe

logger = logging.getLogger(__name__)


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

    def timeframe(self) -> Timeframe | None:
        """Return the timeframe of this feed it has one and is known, otherwise return None."""
        return None

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
                logging.error('Error at playback', exc_info=e)
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
        plt : matplotlib axes
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
        plt = plt.subplot() if hasattr(plt, "subplot") else plt
        times, prices = get_symbol_prices(self, symbol, price_type, timeframe)
        plt.plot(times, prices, **kwargs)
        plt.set_title(symbol)


def get_symbol_prices(
        feed: Feed, symbol: str, price_type="DEFAULT", timeframe: Timeframe | None = None
) -> tuple[list[datetime], list[float]]:
    """Get prices for a single symbol from a feed"""

    x = []
    y = []
    channel = feed.play_background(timeframe)
    while event := channel.get():
        price = event.get_price(symbol, price_type)
        if price:
            x.append(event.time)
            y.append(price)
    return x, y
