from array import array
from datetime import timedelta
from typing import Literal

from roboquant.asset import Asset
from roboquant.event import Event, Bar, TradePrice, Quote
from roboquant.timeframe import Timeframe
from .feed import Feed


class BarAggregatorFeed(Feed):
    """Aggregates Trades or Quotes of another feed into a `Bar` prices.

    When trades are selected, the actual trade prices and volumes are used to create the aggregated bars.
    When quotes are selected, the midpoint prices and no volumes are used to create the aggregated bars.
    """

    def __init__(
        self,
        feed: Feed,
        frequency: timedelta,
        price_type: Literal["trade", "quote"] = "quote",
        send_remaining : bool =False,
        continuation : bool =True,
    ):
        super().__init__()
        self.feed = feed
        self.freq = frequency
        self.send_remaining = send_remaining
        self.continuation = continuation
        self.price_type = price_type

    def __aggr_trade2bar(self, evt: Event, bars: dict[Asset, Bar], freq: str):
        for item in evt.items:
            if self.price_type == "trade" and isinstance(item, TradePrice):
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

    def play(self, timeframe: Timeframe | None = None):
        bars: dict[Asset, Bar] = {}
        next_time = None
        bar_freq = str(self.freq)
        for event in self.feed.play(timeframe):
            if not next_time:
                next_time = event.time + self.freq
            elif event.time >= next_time:
                items = list(bars.values())
                evt = Event(next_time, items)
                yield evt
                bars = {} if not self.continuation else self.__get_continued_bars(bars)
                next_time += self.freq

            self.__aggr_trade2bar(event, bars, bar_freq)

        if bars and self.send_remaining and next_time:
            items = list(bars.values())
            evt = Event(next_time, items)
            yield evt


class TimeGroupingFeed(Feed):
    """Group events that occur closely after each other into a single event. It uses the time of the events to
    determine if they are close to each other.
    """

    def __init__(
        self,
        feed: Feed,
        timeout: float=5.0,
    ):
        super().__init__()
        self.feed = feed
        self.timeout = timeout

    def play(self, timeframe: Timeframe | None = None):
        items = []
        time = None
        for event in self.feed.play(timeframe):
            time = time or event.time
            remaining = self.timeout - (event.time - time).total_seconds()

            if remaining <= 0.0:
                new_event = Event(time, items)
                yield new_event
                items = []
                time = event.time

            items.extend(event.items)

        if time:
            new_event = Event(time, items)
            yield new_event
