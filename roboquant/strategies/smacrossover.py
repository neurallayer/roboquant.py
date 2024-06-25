import collections

import numpy as np

from roboquant.account import Account
from roboquant.event import Event
from roboquant.strategies.basestrategy import BaseStrategy


class SMACrossover(BaseStrategy):
    """SMA Crossover Strategy"""

    def __init__(self, min_period: int = 13, max_period: int = 26):
        super().__init__()
        self._history: dict[str, collections.deque] = {}
        self._prev_ratings: dict[str, bool] = {}
        self.min_period = min_period
        self.max_period = max_period

    def _process(self, symbol: str):
        prices = np.asarray(self._history[symbol])

        # SMA(MIN) > SMA(MAX)
        new_rating: bool = prices[-self.min_period:].mean() > prices[-self.max_period:].mean()
        if symbol in self._prev_ratings:
            prev_rating = self._prev_ratings[symbol]
            if prev_rating != new_rating:
                if new_rating:
                    self.add_buy_order(symbol)
                else:
                    self.add_sell_order(symbol)

        self._prev_ratings[symbol] = new_rating

    def process(self, event: Event, account: Account):
        for (symbol, item) in event.price_items.items():
            h = self._history.get(symbol)

            if h is None:
                maxlen = max(self.max_period, self.min_period)
                h = collections.deque(maxlen=maxlen)
                self._history[symbol] = h

            h.append(item.price())
            if len(h) == h.maxlen:
                self._process(symbol)
