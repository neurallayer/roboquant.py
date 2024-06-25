from abc import abstractmethod

from roboquant.event import Bar
from roboquant.strategies.basestrategy import BaseStrategy
from roboquant.strategies.buffer import OHLCVBuffer


class TaStrategy(BaseStrategy):
    """Abstract base class for other strategies that helps to implement trading solutions
    based on technical indicators using bars.

    Subclasses should implement the _create_signal method. This method is only invoked once
    there is at least `size` history for an individual symbol.
    """

    def __init__(self, size: int) -> None:
        super().__init__()
        self._data: dict[str, OHLCVBuffer] = {}
        self.size = size

    def process(self, event, account):
        for item in event.items:
            if isinstance(item, Bar):
                symbol = item.symbol
                if symbol not in self._data:
                    self._data[symbol] = OHLCVBuffer(self.size)
                ohlcv = self._data[symbol]
                ohlcv.append(item.ohlcv)
                if ohlcv.is_full():
                    self.process_symbol(symbol, ohlcv, item)

    @abstractmethod
    def process_symbol(self, symbol: str, ohlcv: OHLCVBuffer, item: Bar):
        """
        Create zero or more orders for the provided symbol
        """
        ...
