from abc import ABC, abstractmethod
from typing import Iterator

from roboquant.common.asset import Asset
from roboquant.common.event import Event
from roboquant.common.timeframe import Timeframe


class Feed(ABC):
    """
    A Feed represents a source of (financial) events that can be (re-)played to feed a `run`
    with data.

    Although the most common type of events are those containing market data, other types of
    events are also possible. For example, events containing news items or social media posts could
    also be represented as a feed.
    """

    @abstractmethod
    def play(self, timeframe: Timeframe| None = None) -> Iterator[Event]:
        """
        (Re-)play the events contained in the feed.

        Args:
            timeframe: An optional timeframe to limit the events to.

        Returns:
            An iterator of the events.
        """
        ...

    @abstractmethod
    def assets(self) -> set[Asset]:
        """Returns the set of unique assets that are contained in the feed.
        Most feeds will contain events for multiple assets, and this method can be used to retrieve those assets.
        There might be feeds that don't know upfront the assets they contain and they can return an empty set.
        """
        ...
