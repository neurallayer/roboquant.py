from array import array
from datetime import timedelta
from typing import Literal

from roboquant.event import Event, Candle, Trade, Quote
from .eventchannel import EventChannel
from .feed import Feed


class CandleFeed(Feed):
    """Aggregates Trades or Quotes of another feed into candles.

    When trades are used, the actual trade price and volume are used to create aggregated candles.
    When quotes are used, the midpoint price and no volume are used to create aggregated candles.
    """

    def __init__(
        self,
        feed: Feed,
        frequency: timedelta,
        item_type: Literal["trade", "quote"] = "trade",
        send_remaining=False,
        continuation=True,
    ):
        super().__init__()
        self.feed = feed
        self.freq = frequency
        self.send_remaining = send_remaining
        self.continuation = continuation
        self.item_type = item_type

    def __aggr_trade2candle(self, evt: Event, candles: dict[str, Candle], freq: str):

        for item in evt.items:

            if self.item_type == "trade" and isinstance(item, Trade):
                price = item.trade_price
                volume = item.trade_volume
            elif self.item_type == "quote" and isinstance(item, Quote):
                price = item.midpoint_price
                volume = float("nan")
            else:
                continue

            symbol = item.symbol
            candle = candles.get(symbol)
            if candle:
                ohlcv = candle.ohlcv
                ohlcv[3] = price  # close
                if price > ohlcv[1]:
                    ohlcv[1] = price  # high
                if price < ohlcv[2]:
                    ohlcv[2] = price  # low
                ohlcv[4] += volume
            else:
                candles[symbol] = Candle(symbol, array("f", [price, price, price, price, volume]), freq)

    def __get_continued_candles(self, candles: dict[str, Candle]) -> dict[str, Candle]:
        result = {}
        for symbol, item in candles.items():
            p = item.price("CLOSE")
            v = 0.0 if self.item_type == "trade" else float("nan")
            candle = Candle(symbol, array("f", [p, p, p, p, v]))
            result[symbol] = candle
        return result

    def play(self, channel: EventChannel):
        candles: dict[str, Candle] = {}
        src_channel = self.feed.play_background(channel.timeframe, channel.maxsize)
        next_time = None
        candle_freq = str(self.freq)
        while event := src_channel.get():
            if not next_time:
                next_time = event.time + self.freq
            elif event.time >= next_time:
                items = list(candles.values())
                evt = Event(next_time, items)
                channel.put(evt)
                candles = {} if not self.continuation else self.__get_continued_candles(candles)
                next_time += self.freq

            self.__aggr_trade2candle(event, candles, candle_freq)

        if candles and self.send_remaining and next_time:
            items = list(candles.values())
            evt = Event(next_time, items)
            channel.put(evt)
