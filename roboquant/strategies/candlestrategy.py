from abc import abstractmethod, ABC
from typing import Dict

from roboquant.strategies.strategy import Strategy
from roboquant.event import Candle
from roboquant.strategies.buffer import OHLCVBuffer


class CandleStrategy(Strategy, ABC):
    """Abstract base class for other strategies that helps to implement trading solutions
    based on technical indicators using candles.
    """

    def __init__(self, size: int) -> None:
        super().__init__()
        self._data: Dict[str, OHLCVBuffer] = {}
        self.size = size

    def give_ratings(self, event):
        ratings = {}
        for item in event.items:
            if isinstance(item, Candle):
                symbol = item.symbol
                if symbol not in self._data:
                    self._data[symbol] = OHLCVBuffer(self.size)
                ohlcv = self._data[symbol]
                ohlcv.append(item.ohlcv)
                if ohlcv.is_full():
                    rating = self._give_rating(symbol, ohlcv)
                    if rating is not None:
                        ratings[symbol] = rating
        return ratings

    @abstractmethod
    def _give_rating(self, symbol: str, ohlcv: OHLCVBuffer) -> float | None:
        """
        Subclasses should implement this method and return a rating or None for the provided symbol and ohlcv data.
        """
        ...
