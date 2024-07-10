from array import array
from datetime import timedelta
from typing import Literal

from roboquant.asset import Asset
from roboquant.event import Event, Bar, Trade, Quote
from .eventchannel import EventChannel
from .feed import Feed


class AggregatorFeed(Feed):
    """Aggregates Trades or Quotes of another feed into a `Bar` prices.

    When trades are selected, the actual trade prices and volumes are used to create the aggregated bars.
    When quotes are selected, the midpoint prices and no volumes are used to create the aggregated bars.
    """

    def __init__(
        self,
        feed: Feed,
        frequency: timedelta,
        price_type: Literal["trade", "quote"] = "quote",
        send_remaining=False,
        continuation=True,
    ):
        super().__init__()
        self.feed = feed
        self.freq = frequency
        self.send_remaining = send_remaining
        self.continuation = continuation
        self.price_type = price_type

    def __aggr_trade2bar(self, evt: Event, bars: dict[Asset, Bar], freq: str):

        for item in evt.items:

            if self.price_type == "trade" and isinstance(item, Trade):
                price = item.trade_price
                volume = item.trade_volume
            elif self.price_type == "quote" and isinstance(item, Quote):
                price = item.midpoint_price
                volume = float("nan")
            else:
                continue

            symbol = item.asset
            b = bars.get(symbol)
            if b:
                ohlcv = b.ohlcv
                ohlcv[3] = price  # close
                if price > ohlcv[1]:
                    ohlcv[1] = price  # high
                if price < ohlcv[2]:
                    ohlcv[2] = price  # low
                ohlcv[4] += volume
            else:
                bars[symbol] = Bar(symbol, array("f", [price, price, price, price, volume]), freq)

    def __get_continued_bars(self, bars: dict[Asset, Bar]) -> dict[Asset, Bar]:
        result = {}
        for symbol, item in bars.items():
            p = item.price("CLOSE")
            v = 0.0 if self.price_type == "trade" else float("nan")
            b = Bar(symbol, array("f", [p, p, p, p, v]))
            result[symbol] = b
        return result

    def play(self, channel: EventChannel):
        bars: dict[Asset, Bar] = {}
        src_channel = self.feed.play_background(channel.timeframe, channel.maxsize)
        next_time = None
        bar_freq = str(self.freq)
        while event := src_channel.get():
            if not next_time:
                next_time = event.time + self.freq
            elif event.time >= next_time:
                items = list(bars.values())
                evt = Event(next_time, items)
                channel.put(evt)
                bars = {} if not self.continuation else self.__get_continued_bars(bars)
                next_time += self.freq

            self.__aggr_trade2bar(event, bars, bar_freq)

        if bars and self.send_remaining and next_time:
            items = list(bars.values())
            evt = Event(next_time, items)
            channel.put(evt)
