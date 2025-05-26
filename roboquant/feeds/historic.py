from abc import ABC
from datetime import datetime
from itertools import chain

from roboquant.asset import Asset
from roboquant.event import Event, PriceItem
from roboquant.timeframe import Timeframe
from .feed import Feed


class HistoricFeed(Feed, ABC):
    """
    Abstract base class for feeds that produce historic price-items.
    Internally, it uses a sorted-by-datetime dictionary to store the data. So all data
    is kept in memory.
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

    def assets(self) -> list[Asset]:
        """Return the list of unique symbols available in this feed"""
        self._update()
        return list(self.__assets)

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

    def timeline(self) -> list[datetime]:
        """Return the timeline of this feed as a list of datatime objects"""
        self._update()
        return list(self.__data.keys())

    def timeframe(self) -> Timeframe:
        """Return the timeframe of this feed"""
        tl = self.timeline()
        if tl:
            return Timeframe(tl[0], tl[-1], inclusive=True)

        return Timeframe.EMPTY

    def _update(self):
        if self.__modified:
            self.__data = dict(sorted(self.__data.items()))
            price_items = chain.from_iterable(self.__data.values())
            self.__assets = {item.asset for item in price_items}
            self.__modified = False

    def get_first_event(self) -> Event | None:
        """Return the first event in this feed, or None if no events are available"""
        self._update()
        if not self.__data:
            return None

        first_time = next(iter(self.__data.keys()))
        items = self.__data[first_time]
        return Event(first_time, items)

    def get_last_event(self) -> Event | None:
        """Return the last event in this feed, or None if no events are available"""
        self._update()
        if not self.__data:
            return None

        last_time = next(reversed(self.__data.keys()))
        items = self.__data[last_time]
        return Event(last_time, items)

    def play(self, timeframe: Timeframe | None = None):
        self._update()
        for k, v in self.__data.items():
            if not timeframe or k in timeframe:
                yield Event(k, v)
            elif k <= timeframe.start:
                continue
            else:
                break

    def __repr__(self) -> str:
        feed = self.__class__.__name__
        return f"{feed}(assets={len(self.assets())} timeframe={self.timeframe()})"
