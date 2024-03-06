import collections

import numpy as np

from roboquant.signal import Signal
from roboquant.strategies.strategy import Strategy


class SMACrossover(Strategy):
    """SMA Crossover Strategy"""

    def __init__(self, min_period: int = 13, max_period: int = 26):
        super().__init__()
        self._history: dict[str, collections.deque] = {}
        self._prev_ratings: dict[str, bool] = {}
        self.min_period = min_period
        self.max_period = max_period

    def __get_signal(self, symbol: str) -> None | Signal:
        prices = np.asarray(self._history[symbol])

        # SMA(MIN) > SMA(MAX)
        new_rating: bool = prices[-self.min_period:].mean() > prices[-self.max_period:].mean()
        result = None
        if symbol in self._prev_ratings:
            prev_rating = self._prev_ratings[symbol]
            if prev_rating != new_rating:
                result = Signal.buy() if new_rating else Signal.sell()

        self._prev_ratings[symbol] = new_rating
        return result

    def create_signals(self, event):
        signals: dict[str, Signal] = {}
        for (symbol, item) in event.price_items.items():
            h = self._history.get(symbol)

            if h is None:
                maxlen = max(self.max_period, self.min_period)
                h = collections.deque(maxlen=maxlen)
                self._history[symbol] = h

            h.append(item.price())
            if len(h) == h.maxlen:
                if signal := self.__get_signal(symbol):
                    signals[symbol] = signal

        return signals
