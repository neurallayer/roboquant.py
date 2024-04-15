
from roboquant.event import Event
from .eventchannel import EventChannel
from .feed import Feed


class CollectorFeed(Feed):
    """Collect events into one new event if they occur close to eachother.

    Close to eachother is defined by the timeout is seconds. If there is no new
    event in the specified timeout, all previous events will be bundled together and
    put on the channel.
    """

    def __init__(
        self,
        feed: Feed,
        timeout=5.0,
    ):
        super().__init__()
        self.feed = feed
        self.timeout = timeout

    def play(self, channel: EventChannel):
        src_channel = self.feed.play_background(channel.timeframe, channel.maxsize)
        items = []
        while event := src_channel.get(self.timeout):
            if event.is_empty() and items:
                new_event = Event(event.time, items)
                channel.put(new_event)
                items = []

            items.extend(event.items)
