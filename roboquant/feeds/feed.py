from abc import ABC, abstractmethod
from typing import Iterator

from roboquant.event import Event
from roboquant.timeframe import Timeframe


class Feed(ABC):
    """
    A Feed represents a source of (financial) events that can be (re-)played to feed a `run`.

    Although the most common type of events are those containing market data, other types of
    events are also possible. For example, events containing news items or social media posts could
    also be represented as a feed.
    """

    @abstractmethod
    def play(self, timeframe: Timeframe| None = None) -> Iterator[Event]: # Generator[Event, Any, None]:
        """
        (Re-)play the events contained in the feed.

        Args:
            timeframe: An optional timeframe to limit the events to.

        Returns:
            An iterator of the events.
        """
        ...

    def timeframe(self) -> Timeframe:
        """Return the timeframe of this feed, default is Timeframe.INFINITE"""
        return Timeframe.INFINITE
