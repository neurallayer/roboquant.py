from array import array
from datetime import timedelta

from roboquant.event import Event, Candle, Trade
from roboquant.feeds.feedutil import play_background
from .eventchannel import EventChannel
from .feed import Feed


class CandleFeed(Feed):
    """Aggregates Trades of another feed into candles"""

    def __init__(self, feed: Feed, frequency: timedelta, send_remaining=False, continuation=True):
        super().__init__()
        self.feed = feed
        self.freq = frequency
        self.send_remaining = send_remaining
        self.continuation = continuation

    @staticmethod
    def __aggr_trade2candle(evt: Event, candles: dict[str, Candle]):
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

    @staticmethod
    def __get_continued_candles(candles: dict[str, Candle]) -> dict[str, Candle]:
        result = {}
        for symbol, item in candles.items():
            p = item.price("CLOSE")
            candle = Candle(symbol, array("f", [p, p, p, p, 0.0]))
            result[symbol] = candle
        return result

    def play(self, channel: EventChannel):
        src_channel = channel.copy()
        candles: dict[str, Candle] = {}
        play_background(self.feed, src_channel)
        next_time = None
        while event := src_channel.get():
            if not next_time:
                next_time = event.time + self.freq
            elif event.time >= next_time:
                items = list(candles.values())
                evt = Event(next_time, items)
                channel.put(evt)
                candles = {} if not self.continuation else self.__get_continued_candles(candles)
                next_time += self.freq

            self.__aggr_trade2candle(event, candles)

        if candles and self.send_remaining and next_time:
            items = list(candles.values())
            evt = Event(next_time, items)
            channel.put(evt)
