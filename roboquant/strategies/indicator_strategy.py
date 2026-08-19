from abc import abstractmethod
from typing import override

from roboquant.common.asset import Asset
from roboquant.common.event import Event
from roboquant.common.signal import Signal
from roboquant.util.buffer import AssetBuffers, OHLCVBuffer
from roboquant.strategies.strategy import Strategy


class IndicatorStrategy(Strategy):
    """Abstract base class for other strategies that helps to implement trading solutions
    based on technical indicators using a history of bars (aka candlesticks).

    Subclasses should implement the `process_asset` method. This method is only invoked once
    there is at least the `size` history for an individual asset available.
    """

    def __init__(self, period: int) -> None:
        super().__init__()
        self._data = AssetBuffers(period)
        self.period = period

    @override
    def create_signals(self, event: Event) -> list[Signal]:
        result: list[Signal] = []
        assets = self._data.add_event(event)
        for asset in assets:
            ohlcv = self._data[asset]
            if signal := self._create_signal(asset, ohlcv):
                result.append(signal)
        return result

    @abstractmethod
    def _create_signal(self, asset: Asset, ohlcv: OHLCVBuffer) -> Signal | None:
        """
        Create a signal for the provided asset or return None if no signal should be created.
        Subclasses should implement this method. This method is only invoked if enough data
        is available for the asset, i.e. the size of the OHLCVBuffer is at least `self.period`.

        Example:
        ```
        prices = ohlcv.close()
        if prices[-10:].mean() > prices[-20:].mean():
            return Signal.buy(asset)
        ```
        """
        ...


class MultiAssetIndicatorStrategy(Strategy):
    """Abstract base class for other strategies that helps to implement trading solutions
    based on technical indicators using a history of bars (aka candlesticks).

    Compared to the `TaStrategy`, this class is designed to work with multiple assets at the same time.
    So you can create signals based on the history of multiple assets.

    Subclasses should implement the `process_assets` method. This method is only invoked once
    there is at least one asset that has `size` data available.
    """

    def __init__(self, period: int) -> None:
        super().__init__()
        self._data = AssetBuffers(period)
        self.period = period

    @override
    def create_signals(self, event: Event) -> list[Signal]:
        assets = self._data.add_event(event)
        if assets:
            data = {asset: self._data[asset] for asset in assets}
            return self.process_assets(data)
        return []

    @abstractmethod
    def process_assets(self, data: dict[Asset, OHLCVBuffer]) -> list[Signal]:
        """
        Create zero or more signals for the provided assets or return an empty list if no signal is created.
        Subclasses should implement this method.

        This method is only invoked with the assets that have already enough data available, i.e. the size
        of the OHLCVBuffer is at least `self.period`.
        """
        ...
