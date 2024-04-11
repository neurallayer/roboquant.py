from array import array
import random
import string
from datetime import datetime, timedelta, timezone
from typing import Literal

import numpy as np

from roboquant.event import Bar, Trade
from .historic import HistoricFeed


class RandomWalk(HistoricFeed):
    """This feed simulates the random-walk of stock prices.
    It can generate trade or bar prices."""

    def __init__(
        self,
        n_symbols: int = 10,
        n_prices: int = 1_000,
        item_type: Literal["bar", "trade"] = "bar",
        start_date: str | datetime = "2020-01-01T00:00:00+00:00",
        frequency=timedelta(days=1),
        start_price_min: float = 50.0,
        start_price_max: float = 200.0,
        volume: float = 1000.0,
        stdev=0.01,
        seed=None,
        symbol_len=4,
    ):
        super().__init__()
        rnd = np.random.default_rng(seed)
        symbols = self.__get_symbols(rnd, n_symbols, symbol_len)
        assert len(symbols) == n_symbols

        start_date = start_date if isinstance(start_date, datetime) else datetime.fromisoformat(start_date)
        start_date = start_date.astimezone(timezone.utc)
        timeline = [start_date + frequency * i for i in range(n_prices)]

        item_gen = self.__get_bar if item_type == "bar" else self.__get_trade

        for symbol in symbols:
            prices = self.__price_path(rnd, n_prices, stdev, start_price_min, start_price_max)
            for i in range(n_prices):
                item = item_gen(symbol, prices[i], volume)
                self._add_item(timeline[i], item)

    @staticmethod
    def __get_trade(symbol, price, volume):
        return Trade(symbol, price, volume)

    @staticmethod
    def __get_bar(symbol, price, volume):
        high = price * (1.0 + abs(random.gauss(mu=0.0, sigma=1.0)))
        low = price * (1.0 - abs(random.gauss(mu=0.0, sigma=1.0)))
        close = random.uniform(low, high)
        return Bar(symbol, array("f", [price, high, low, close, volume]))

    @staticmethod
    def __get_symbols(
        rnd,
        n_symbols,
        symbol_len,
    ):
        symbols = set()
        alphabet = np.array(list(string.ascii_uppercase))
        while len(symbols) < n_symbols:
            symbol = "".join(rnd.choice(alphabet, size=symbol_len))
            symbols.add(symbol)
        return symbols

    @staticmethod
    def __price_path(rnd, n, scale, min_price, max_price):
        change = rnd.normal(loc=1.0, scale=scale, size=(n,))
        change[0] = rnd.uniform(min_price, max_price)
        price = change.cumprod()
        return price
