from array import array
import random
import string
from datetime import datetime, timedelta, timezone
from typing import Literal

import numpy as np

from roboquant.asset import Asset, Stock
from roboquant.event import Bar, Trade, Quote
from .historic import HistoricFeed


class RandomWalk(HistoricFeed):
    """This feed simulates the random-walk of stock prices.
    It can generate `Trade`, `Quote`, or `Bar` prices."""

    def __init__(
        self,
        n_symbols: int = 10,
        n_prices: int = 1_000,
        price_type: Literal["bar", "trade", "quote"] = "bar",
        start_date: str | datetime = "2020-01-01T00:00:00+00:00",
        frequency=timedelta(days=1),
        start_price_min: float = 50.0,
        start_price_max: float = 200.0,
        volume: float = 1000.0,
        price_dev=0.01,
        spread_dev=0.001,
        seed=None,
        symbol_len=4,
    ):
        # pylint: disable=too-many-locals
        super().__init__()
        rnd = np.random.default_rng(seed)
        assets = self.__get_assets(rnd, n_symbols, symbol_len)
        assert len(assets) == n_symbols

        start_date = start_date if isinstance(start_date, datetime) else datetime.fromisoformat(start_date)
        start_date = start_date.astimezone(timezone.utc)
        timeline = [start_date + frequency * i for i in range(n_prices)]

        match price_type:
            case "bar":
                item_gen = self.__get_bar
            case "trade":
                item_gen = self.__get_trade
            case "quote":
                item_gen = self.__get_quote
            case _:
                raise ValueError("unsupported item_type", price_type)

        for asset in assets:
            prices = self.__price_path(rnd, n_prices, price_dev, start_price_min, start_price_max)
            for i in range(n_prices):
                item = item_gen(asset, prices[i], volume, spread_dev)
                self._add_item(timeline[i], item)
        self._update()

    @staticmethod
    def __get_trade(symbol, price, volume, _):
        return Trade(symbol, price, volume)

    @staticmethod
    def __get_bar(asset, price, volume, spread_dev):
        high = price * (1.0 + abs(random.gauss(mu=0.0, sigma=spread_dev)))
        low = price * (1.0 - abs(random.gauss(mu=0.0, sigma=spread_dev)))
        close = random.uniform(low, high)
        prices = array("f", [price, high, low, close, volume])
        return Bar(asset, prices)

    @staticmethod
    def __get_quote(symbol, price, volume, spread_dev):
        spread = abs(random.gauss(mu=0.0, sigma=spread_dev)) * price / 2.0
        ask = price + spread
        bid = price - spread
        return Quote(symbol, array("f", [price, ask, volume, bid, volume]))

    @staticmethod
    def __get_assets(
        rnd,
        n_symbols,
        symbol_len,
    ) -> list[Asset]:
        assets = set()
        alphabet = np.array(list(string.ascii_uppercase))
        while len(assets) < n_symbols:
            symbol = "".join(rnd.choice(alphabet, size=symbol_len))
            asset = Stock(symbol)
            assets.add(asset)
        return list(assets)

    @staticmethod
    def __price_path(rnd, n, scale, min_price, max_price):
        change = rnd.normal(loc=1.0, scale=scale, size=(n,))
        change[0] = rnd.uniform(min_price, max_price)
        price = change.cumprod()
        return price
