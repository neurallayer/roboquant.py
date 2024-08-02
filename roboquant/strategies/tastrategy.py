from abc import abstractmethod

from roboquant.asset import Asset
from roboquant.event import Bar
from roboquant.signal import Signal
from roboquant.strategies.buffer import OHLCVBuffer
from roboquant.strategies.strategy import Strategy


class TaStrategy(Strategy):
    """Abstract base class for other strategies that helps to implement trading solutions
    based on technical indicators using a history of bars/candlesticks.

    Subclasses should implement the `process_asset` method. This method is only invoked once
    there is at least `size` history for an individual asset.
    """

    def __init__(self, size: int) -> None:
        super().__init__()
        self._data: dict[Asset, OHLCVBuffer] = {}
        self.size = size

    def create_signals(self, event):
        result = []
        for item in event.items:
            if isinstance(item, Bar):
                asset = item.asset
                if asset not in self._data:
                    self._data[asset] = OHLCVBuffer(self.size)
                ohlcv = self._data[asset]
                if ohlcv.append(item.ohlcv):
                    if signal := self.process_asset(asset, ohlcv):
                        result.append(signal)
        return result

    @abstractmethod
    def process_asset(self, asset: Asset, ohlcv: OHLCVBuffer) -> Signal | None:
        """
        Create zero or more orders for the provided symbol
        """
        ...
