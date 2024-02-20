from array import array
from .feed import Feed
from roboquant.event import Event, Candle, Trade
from roboquant.feeds.feedutil import play_background
from .eventchannel import EventChannel

from datetime import timedelta


class CandleFeed(Feed):
    """Aggregates Trades of another feed into Candles"""

    def __init__(self, feed: Feed, frequency: timedelta, send_remaining=False):
        super().__init__()
        self.feed = feed
        self.freq = frequency
        self.send_remaining = send_remaining

    def __aggr_trade2candle(self, evt: Event, candles: dict[str, Candle]):
        for item in evt.items:
            if isinstance(item, Trade):
                symbol = item.symbol
                p = item.trade_price
                candle = candles.get(symbol)
                if candle:
                    ohlcv = candle.ohlcv
                    ohlcv[3] = p  # close
                    if p > ohlcv[1]:
                        ohlcv[1] = p  # high
                    if p < ohlcv[2]:
                        ohlcv[2] = p  # low
                    ohlcv[4] += item.trade_volume
                else:
                    candles[symbol] = Candle(symbol, array("f", [p, p, p, p, item.trade_volume]))

    def play(self, channel: EventChannel):
        src_channel = channel.copy()
        candles: dict[str, Candle] = {}
        play_background(self.feed, src_channel)
        next = None
        while event := src_channel.get():
            if not next:
                next = event.time + self.freq
            elif event.time >= next:
                items = list(candles.values())
                evt = Event(next, items)
                channel.put(evt)
                candles = {}
                next += self.freq

            self.__aggr_trade2candle(event, candles)

        if candles and self.send_remaining and next:
            items = list(candles.values())
            evt = Event(next, items)
            channel.put(evt)
