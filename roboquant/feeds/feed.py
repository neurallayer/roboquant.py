import threading
from abc import ABC, abstractmethod

from roboquant.feeds.eventchannel import EventChannel, ChannelClosed
from roboquant.timeframe import Timeframe


class Feed(ABC):
    """A feed (re-)plays events"""

    @abstractmethod
    def play(self, channel: EventChannel):
        """(re-)play the events in the feed and put them on the provided event channel"""
        ...

    def play_background(self, timeframe: Timeframe | None = None, channel_capacity: int = 10) -> EventChannel:
        """Play this feed in the background on its own thread.
        The returned event-channel will be closed after the playing has finished.
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

    def plot(self, plt, *symbols: str, price_type: str = "DEFAULT", timeframe: Timeframe | None = None, **kwargs):
        """Plot the prices of one or more symbols"""
        channel = self.play_background(timeframe)
        result = {}
        while evt := channel.get():
            for symbol, price in evt.get_prices(price_type).items():
                if symbols and symbol not in symbols:
                    continue
                if symbol not in result:
                    result[symbol] = ([], [])
                data = result[symbol]
                data[0].append(evt.time)
                data[1].append(price)

        for symbol, data in result.items():
            plt.plot(data[0], data[1], **kwargs)
            if hasattr(plt, "set_title"):
                # assume we are in a subplot
                plt.set_title(symbol)
            else:
                plt.title(symbol)
                plt.show()


