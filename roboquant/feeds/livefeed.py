from datetime import datetime, timedelta
from queue import SimpleQueue
from queue import Empty, Full
from typing import Iterator, override
import logging

from roboquant.common.asset import Asset
from roboquant.common.event import Event
from roboquant.common.timeframe import Timeframe, utcnow
from .feed import Feed


logger = logging.getLogger(__name__)


class LiveFeed(Feed):
    """
    Abstract base class for feeds that produce live price-items.

    It will ensure that events that are published are monotonic
    in time (so always increasing). If a new event has a timestamp that is
    before or equal to the previous event, the timestamp will be autocorrected
    so the event occurs after the previous event. The default is to increment it by
    1 microsecond over the previous event, but this is configurable.

    There is also support for creating an empty event (heartbeat) if for a certain duration no event was
    received.
    """

    def __init__(self):
        super().__init__()
        self.__queue: SimpleQueue[Event] | None = None
        self.__last_time = datetime.fromisoformat("1900-01-01T00:00:00+00:00")
        self.increment = timedelta(microseconds=1)
        self.heartbeat_timeout: float = 10.0
        self._registry: dict[str, Asset] = {}

    @override
    def play(self, timeframe: Timeframe | None = None) -> Iterator[Event]:
        queue = SimpleQueue()
        self.__queue = queue
        timeout = self.heartbeat_timeout
        while True:
            try:
                if event := queue.get(timeout=timeout):
                    if not timeframe or event.time in timeframe:
                        yield event
                    elif event.time < timeframe.start:
                        continue
                    else:
                        break
            except Empty:
                # We are here due to a timeout, so we need to send a heartbeat event
                time = utcnow()
                if not timeframe or time in timeframe:
                    yield Event.empty(time)
                elif time < timeframe.start:
                    continue
                else:
                    break

        self.__queue = None

    def _put(self, event: Event):
        """Put an event onto the queue.

        Subclasses should call this method to publish an event. It also takes care of:
        - ensuring that the events are monotonic in time. If the event is not monotonic in time, it will be corrected.
        - ensuring that the event is not too far in the future. If it is, it will be corrected to the current time.
        """
        if self.__queue:
            try:
                now = utcnow()
                if event.time > now + timedelta(seconds=60):
                    logger.warning(
                        "event %s is more than 60 seconds in the future, using current time instead", event
                    )
                    event.time = now
                if event.time <= self.__last_time:
                    event.time = self.__last_time + self.increment
                self.__last_time = event.time
                self.__queue.put(event)
            except Full:
                pass

    def register(self, symbol: str, asset: Asset):
        """Register an asset for a symbol name"""
        self._registry[symbol] = asset

    def get_asset(self, symbol: str) -> Asset | None:
        """Get the asset (if any) for the provided symbol name"""
        return self._registry.get(symbol)

    @override
    def assets(self) -> list[Asset]:
        return list(self._registry.values())
