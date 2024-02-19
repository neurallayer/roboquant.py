from typing import Protocol

from .eventchannel import EventChannel


class Feed(Protocol):
    """A feed (re-)plays events"""

    def play(self, channel: EventChannel):
        """(re-)play the events in the feed and put them on the provided event channel"""
        ...
