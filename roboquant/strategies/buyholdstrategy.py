from typing import override

from roboquant.common.event import Event
from roboquant.common.signal import Signal
from roboquant.strategies.strategy import Strategy


class BuyHoldStrategy(Strategy):
    """Create buy signals for the assets found in the events.
    You can compare a custom strategy to this one to see possible
    differences in performance.
    """

    def __init__(self, wait: int = 0) -> None:
        super().__init__()
        self.__wait = wait

    @override
    def create_signals(self, event: Event) -> list[Signal]:
        if self.__wait > 0:
            self.__wait -= 1
            return []

        return [
            Signal.buy(asset) for asset in event.price_items.keys()
        ]
