from datetime import datetime, timedelta, timezone
from multiprocessing import Queue
from queue import Empty, Full
from typing import Any, Generator

from roboquant.event import Event
from roboquant.timeframe import Timeframe
from .feed import Feed


class LiveFeed(Feed):
    """
    Abstract base class for feeds that produce live price-items. It will ensure that
    events that are published are monotonic in time (so always increasing).

    If a new event has a timestamp that is before or equal to the previous event, the
    timestamp will be corrected so the event occurs after the previous event.

    The default is to increment it by 1 microsecond over the previous event, but this is configurable.
    """

    def __init__(self):
        super().__init__()
        self._queue: Queue | None = None
        self._last_time = datetime.fromisoformat("1900-01-01T00:00:00+00:00")
        self.increment = timedelta(microseconds=1)
        self.heartbeat_timeout = 10

    def play(self, timeframe: Timeframe | None = None) -> Generator[Event, Any, None]:
        self._queue = Queue()
        timeout = self.heartbeat_timeout
        while True:
            try:
                if event := self._queue.get(timeout=timeout):
                    if not timeframe or event.time in timeframe:
                        yield event
                    elif event.time < timeframe.start:
                        continue
                    else:
                        break
            except Empty:
                # We are here due to a timeout, so we need to send a heartbeat event
                event = Event(datetime.now(tz=timezone.utc), [])
                if not timeframe or event.time in timeframe:
                    yield event
                elif event.time < timeframe.start:
                    continue
                else:
                    break

        self._queue.close()
        self._queue = None

    def _put(self, event: Event):
        """Put an event on the queue. If the event is not monotonic in time, it will be corrected.
        Subclasses should call this method to publish new live events.
        """
        if self._queue:
            try:
                if event.time <= self._last_time:
                    event.time = self._last_time + self.increment
                self._last_time = event.time
                self._queue.put(event)
            except Full:
                pass
