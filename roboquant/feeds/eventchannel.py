from datetime import datetime, timezone
from queue import Empty, Queue

from roboquant.event import Event
from roboquant.timeframe import Timeframe


class ChannelClosed(Exception):
    """Thrown when the channel is already closed and trying to put a new event on the channel."""


class EventChannel:
    """Event channel is used as a communication channel between a `Feed` and the `Roboquant` engine.
    The `Feed` puts events on the channel, and these are consumed and further processed by the `Roboquant` engine.
    """

    def __init__(self, timeframe: Timeframe | None = None, maxsize: int = 10):
        """
        Args:

        - timeframe: limit events to this timeframe and ignore the other ones
        - size: the maxsize of events stored in the channel
        """

        self.timeframe = timeframe
        self._queue = Queue(maxsize)
        self._closed = False

    @property
    def maxsize(self):
        """The maximum size of the queue"""
        return self._queue.maxsize

    def put(self, event: Event):
        """Put an event on this channel.

        - If the event is outside the timeframe of this channel, it will be ignored.
        - If the event is after the timeframe of this channel, the channel will be closed.
        - If the channel is already closed, a ChannelClosed exception will be raised.
        """
        if self._closed:
            raise ChannelClosed()

        if self.timeframe is None or event.time in self.timeframe:
            self._queue.put(event)
        elif not event.time < self.timeframe.start:
            # we get in this branch when timeframe is not None and
            # the event is past the provided timeframe.
            self.close()

    def get(self, timeout: float|None=None) -> Event | None:
        """Returns the next event or None if this channel is just closed.

        - timeout: the timeout in seconds to send a heartbeat in case no other event was received, default is None.
        A heartbeat is an empty event.
        """
        try:
            return self._queue.get(timeout=timeout)
        except Empty:
            now = datetime.now(timezone.utc)
            if self.timeframe is None or now in self.timeframe:
                return Event.empty(now)

            self._closed = True
        return None

    def close(self):
        """close this channel and put a None message on the channel to indicate to consumers it is closed"""
        if not self._closed:
            self._closed = True
            self._queue.put(None)

    def copy(self):
        """Create a copy of this channel. Items on the original channel will not be copied to the new channel."""
        return EventChannel(self.timeframe, self._queue.maxsize)

    @property
    def is_closed(self) -> bool:
        """Returns true if this channel is closed and no more new events will be put on this channel"""
        return self._closed
