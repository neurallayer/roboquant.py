import string
from datetime import datetime, timedelta, timezone

import numpy as np

from roboquant.event import Trade
from .historic import HistoricFeed


class RandomWalk(HistoricFeed):
    """This feed simulates the random-walk of stock prices."""

    def __init__(
            self,
            n_symbols=10,
            n_prices=1_000,
            start_date: str | datetime = "2020-01-01T00:00:00+00:00",
            frequency=timedelta(days=1),
            start_price_min=50.0,
            start_price_max=200.0,
            volume=1000.0,
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

        for symbol in symbols:
            prices = self.__price_path(rnd, n_prices, stdev, start_price_min, start_price_max)
            for i in range(n_prices):
                item = Trade(symbol, prices[i], volume)
                self._add_item(timeline[i], item)

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
