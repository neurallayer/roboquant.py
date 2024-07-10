from abc import ABC
from datetime import datetime
from itertools import chain

from roboquant.asset import Asset
from roboquant.event import Event, PriceItem
from roboquant.timeframe import EMPTY_TIMEFRAME, Timeframe
from .eventchannel import EventChannel
from .feed import Feed
from .feedutil import get_ohlcv


class HistoricFeed(Feed, ABC):
    """
    Abstract base class for feeds that produce historic price-items.
    """

    def __init__(self):
        super().__init__()
        self.__data: dict[datetime, list[PriceItem]] = {}
        self.__modified = False
        self.__assets: set[Asset] = set()

    def _add_item(self, time: datetime, item: PriceItem):
        """Add a price-item at a moment in time to this feed.
        Subclasses should invoke this method to populate the historic-feed.

        Items added at the same time, will be part of the same event.
        So each unique time will only produce a single event.
        """

        self.__modified = True

        if time not in self.__data:
            self.__data[time] = [item]
        else:
            items = self.__data[time]
            items.append(item)

    @property
    def assets(self):
        """Return the list of unique symbols available in this feed"""
        self.__update()
        return self.__assets

    @property
    def events(self):
        """Return the total number of events"""
        return len(self.__data)

    def timeline(self) -> list[datetime]:
        """Return the timeline of this feed as a list of datatime objects"""
        self.__update()
        return list(self.__data.keys())

    def timeframe(self):
        """Return the timeframe of this feed"""
        tl = self.timeline()
        if tl:
            return Timeframe(tl[0], tl[-1], inclusive=True)

        return EMPTY_TIMEFRAME

    def get_ohlcv(self, asset: Asset, timeframe=None) -> dict[str, list]:
        """Get the OHLCV values for a symbol for the (optional) provided timeframe.
        This makes it easy to plot prices and use them in a dataframe.
        """
        return get_ohlcv(self, asset, timeframe)

    def __update(self):
        if self.__modified:
            self.__data = dict(sorted(self.__data.items()))
            price_items = chain.from_iterable(self.__data.values())
            self.__assets = {item.asset for item in price_items}
            self.__modified = False

    def play(self, channel: EventChannel):
        self.__update()
        for k, v in self.__data.items():
            evt = Event(k, v)
            channel.put(evt)

    def __repr__(self) -> str:
        feed = self.__class__.__name__
        return f"{feed}(events={self.events} assets={len(self.assets)} timeframe={self.timeframe()})"
