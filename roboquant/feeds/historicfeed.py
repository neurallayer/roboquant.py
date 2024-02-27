from abc import ABC
from datetime import datetime
from itertools import chain
from typing import List

from roboquant.event import Event, PriceItem
from roboquant.timeframe import Timeframe
from .eventchannel import EventChannel
from .feed import Feed


class HistoricFeed(Feed, ABC):
    """
    Abstract base class for other feeds that produce historic price-items.
    """

    def __init__(self):
        super().__init__()
        self.__data: dict[datetime, list[PriceItem]] = {}
        self.__modified = False
        self.__symbols = []

    def _add_item(self, time: datetime, item: PriceItem):
        """Add a price-item at a moment in time to this feed"""

        self.__modified = True

        if time not in self.__data:
            self.__data[time] = [item]
        else:
            items = self.__data[time]
            items.append(item)

    @property
    def symbols(self):
        """Return all the symbols available in this feed"""
        self.__update()
        return self.__symbols

    def timeline(self) -> List[datetime]:
        """Return the timeline of this feed"""
        self.__update()
        return list(self.__data.keys())

    def timeframe(self):
        """Return the timeframe of this feed"""
        tl = self.timeline()
        if len(tl) == 0:
            return Timeframe.empty()
        else:
            return Timeframe(tl[0], tl[-1], inclusive=True)

    def __update(self):
        if self.__modified:
            self.__data = dict(sorted(self.__data.items()))
            price_items = chain.from_iterable(self.__data.values())
            self.__symbols = list({item.symbol for item in price_items})
            self.__modified = False

    def play(self, channel: EventChannel):
        self.__update()
        for k, v in self.__data.items():
            evt = Event(k, v)
            channel.put(evt)
