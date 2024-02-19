import collections
import numpy as np
from roboquant.event import Event
from roboquant.strategies.strategy import Strategy


class SMACrossover(Strategy):
    """SMA Crossover Strategy"""

    def __init__(self, min_period: int = 13, max_period: int = 26):
        super().__init__()
        self._history: dict[str, collections.deque] = {}
        self._last_rating: dict[str, bool] = {}
        self.min_period = min_period
        self.max_period = max_period

    def _check_condition(self, symbol: str) -> None | float:
        prices = np.asarray(self._history[symbol])

        # SMA(MIN) > SMA(MAX)
        new_rating: bool = prices[-self.min_period:].mean() > prices[-self.max_period:].mean()
        result = None
        if symbol in self._last_rating:
            last_rating = self._last_rating[symbol]
            if last_rating != new_rating:
                result = 1.0 if last_rating else -1.0

        self._last_rating[symbol] = new_rating
        return result

    def give_ratings(self, event: Event) -> dict[str, float]:
        ratings: dict[str, float] = {}
        for (symbol, item) in event.price_items.items():
            h = self._history.get(symbol)

            if h is None:
                h = collections.deque(maxlen=self.max_period)
                self._history[symbol] = h

            h.append(item.price())
            if len(h) == h.maxlen:
                if rating := self._check_condition(symbol):
                    ratings[symbol] = rating

        return ratings
