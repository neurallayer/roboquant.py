from abc import abstractmethod

from roboquant.asset import Asset
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
        self._data: dict[Asset, OHLCVBuffer] = {}
        self.size = size

    def process(self, event, account):
        for item in event.items:
            if isinstance(item, Bar):
                asset = item.asset
                if asset not in self._data:
                    self._data[asset] = OHLCVBuffer(self.size)
                ohlcv = self._data[asset]
                ohlcv.append(item.ohlcv)
                if ohlcv.is_full():
                    self.process_asset(asset, ohlcv)

    @abstractmethod
    def process_asset(self, asset: Asset, ohlcv: OHLCVBuffer):
        """
        Create zero or more orders for the provided symbol
        """
        ...
