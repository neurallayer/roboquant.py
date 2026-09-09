from abc import abstractmethod
from decimal import Decimal
from roboquant.common.monetary import Currency
from roboquant.common.asset import Stock, Crypto, Forex, Option
from array import array
from datetime import timedelta
from typing import Any, Iterator, Literal, override

import pandas as pd

from roboquant.common.asset import Asset
from roboquant.common.event import Bar, Event, Quote, TradePrice
from roboquant.common.timeframe import Timeframe
from roboquant.feeds.feed import Feed


class BarAggregatorFeed(Feed):
    """Aggregates Trades or Quotes of another feed into a `Bar` prices.

    When trades are selected, the actual trade prices and volumes are used to create the aggregated bars.
    When quotes are selected, the midpoint prices and no volumes are used to create the aggregated bars.
    """

    def __init__(
        self,
        feed: Feed,
        frequency: timedelta | str,
        price_type: Literal["trade", "quote"] = "quote",
        send_remaining : bool =False,
        continuation : bool =True,
    ):
        super().__init__()
        if isinstance(frequency, str):
            frequency = pd.to_timedelta(frequency)
        self.feed = feed
        self.freq = frequency
        self.send_remaining = send_remaining
        self.continuation = continuation
        self.price_type = price_type

    def __aggregate_trade2bar(self, evt: Event, bars: dict[Asset, Bar], freq: str):
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
        result: dict[Asset, Bar] = {}
        for symbol, item in bars.items():
            p = item.price("CLOSE")
            v = 0.0 if self.price_type == "trade" else float("nan")
            b = Bar(symbol, array("f", [p, p, p, p, v]))
            result[symbol] = b
        return result

    @override
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

            self.__aggregate_trade2bar(event, bars, bar_freq)

        if bars and self.send_remaining and next_time:
            items = list(bars.values())
            evt = Event(next_time, items)
            yield evt

    @override
    def assets(self) -> list[Asset]:
        return self.feed.assets()


class TimeGroupingFeed(Feed):
    """Group events that occur closely after each other into a single event.
    It uses the time of the events to determine if they are close to each other.

    Typically used if you have a live feed and for example the 5 minute `Bar`
    prices for different assets arrive just few seconds after each other. This
    allows you to group them still together in a single event.
    """

    def __init__(
        self,
        feed: Feed,
        timeout: float=5.0,
    ):
        super().__init__()
        self.feed = feed
        self.timeout = timeout

    @override
    def play(self, timeframe: Timeframe | None = None) -> Iterator[Event]:
        items: list[Any] = []
        time = None
        last_time = None
        for event in self.feed.play(timeframe):
            send_heartbeat = True
            if items:
                assert time
                remaining = self.timeout - (event.time - time).total_seconds()
                if remaining <= 0.0:
                    yield Event(event.time, items)
                    send_heartbeat = False
                    items = []
            elif event.items:
                time = event.time

            if event.items:
                items.extend(event.items)
            elif send_heartbeat:
                # heart beat event is always propagated unless we already
                # send an event for this time
                yield event
            last_time = event.time

        if last_time and items:
            yield Event(last_time, items)


    @override
    def assets(self) -> list[Asset]:
        return self.feed.assets()



class AssetSerializer:
    """Class for feeds that require (de-)serializing assets
    """

    def __init__(self):
        self.__cache: dict[str, Asset] = {}

    def _serialize_asset(self, asset: Asset) -> str:
        """Serialize an asset to a string representation.

        Args:
            asset (Asset): The asset to serialize.

        Returns:
            The serialized string representation of the asset.
        """

        match asset:
            case Stock() | Option():
                result = f"{asset.asset_class}{asset.symbol}{asset.currency}"
            case Forex() | Crypto():
                result = f"{asset.asset_class}{asset.symbol}{asset.currency}{asset.contract_size}"
            case _:
                raise ValueError(f"unsupported asset type {type(asset)}")

        return result

    def _deserialize_to_asset(self, value: str) -> Asset:
        """Based on the provided string value, deserialize it to the corresponding asset.
        The asset class needs to be registered first using the `register_asset_class` method.

        Under the hood is uses caching to improve performance when repeatedly deserializing the same asset strings.

        Args:
            The serialized string representation of the asset.

        Returns:
            The deserialized asset.
        """
        asset = self.__cache.get(value)
        if not asset:
            asset_class, symbol, code, *args = value.split("")
            currency = Currency(code)
            match asset_class:
                case "Stock":
                    asset = Stock(symbol, currency)
                case "Crypto":
                    asset = Crypto(symbol, currency, None, Decimal(args[-1]))
                case "Forex":
                    asset = Forex(symbol, currency, None, Decimal(args[-1]))
                case "Option":
                    asset = Option(symbol, currency)
                case _:
                    raise ValueError(f"unsupported asset class {value}")
            self.__cache[value] = asset
        return asset


class AssetRegistry:
    """@TODO better implement"""

    def __init__(self):
        self.__registry: dict[str, Asset] = {}

    def register(self, symbol: str, asset: Asset):
        """Register an asset for a symbol name"""
        self.__registry[symbol] = asset

    def _get_asset(self, symbol: str, *args: Any) -> Asset:
        """Get the asset (if any) for the provided symbol name"""
        asset = self.__registry.get(symbol)
        if asset is None:
            asset = self._create_asset(symbol, *args)
            self.__registry[symbol] = asset
        return asset

    @abstractmethod
    def _create_asset(self, symbol: str, *args: Any) -> Asset:
        ...
