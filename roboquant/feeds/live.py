from datetime import datetime, timedelta
import time

from roboquant.event import Event
from .eventchannel import ChannelClosed, EventChannel
from .feed import Feed


class LiveFeed(Feed):
    """
    Abstract base class for feeds that produce live price-items. It will ensure that
    events that are published are monotonic in time (so always increasing).

    If a new event has a timestamp that is before or equal to the previous event, the
    timstamp will be corrected so the event occurs after the previous event.

    The default is to increment it by 1 microsecond over the previous event, but this is configurable.
    """

    def __init__(self):
        super().__init__()
        self._channel = None
        self._last_time = datetime.fromisoformat("1900-01-01T00:00:00+00:00")
        self.increment = timedelta(microseconds=1)

    def play(self, channel: EventChannel):
        self._channel = channel
        while not channel.is_closed:
            time.sleep(1)
        self._channel = None

    def put(self, event: Event):
        if self._channel:
            try:
                if event.time <= self._last_time:
                    event.time = self._last_time + self.increment
                self._channel.put(event)
                self._last_time = event.time
            except ChannelClosed:
                pass
