from datetime import datetime
import logging
from roboquant.event import Event
from roboquant.feeds.feed import Feed
from roboquant.signal import Signal
from roboquant.strategies.strategy import Strategy
from roboquant.timeframe import Timeframe

logger = logging.getLogger(__name__)

class CachedStrategy(Strategy):
    """Cache the signal results of another strategy, usefull for shorter back tests.
    Rather than re-calculating the signals at a certain time, they will be cached and
    replayed once that same time appears again.
    """

    def __init__(self, feed: Feed, strategy: Strategy, timeframe: Timeframe | None = None):
        super().__init__()
        cache: dict[datetime, list[Signal]] = {}
        for event in feed.play(timeframe):
            assert event.time not in cache, "feed has to be monotonic in time"
            signals = strategy.create_signals(event)
            cache[event.time] = signals
        self.__cache = cache

    def timeframe(self):
        if self.__cache:
            timeline = list(self.__cache.keys())
            return Timeframe(timeline[0], timeline[-1], True)
        else:
            return Timeframe.EMPTY

    def create_signals(self, event: Event) -> list[Signal]:
        result = self.__cache.get(event.time)
        if result is None:
            logging.warning("received non-cached timestamp %s, returning no signals", event.time)
            return []
        return result

