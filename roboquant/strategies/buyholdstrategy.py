from typing import override

from roboquant.common.event import Event
from roboquant.common.signal import Signal
from roboquant.strategies.strategy import Strategy


class BuyHoldStrategy(Strategy):
    """A simple buy-and-hold strategy.

    It generates a buy signal for every asset it encounters in an event.
    Use it as a baseline to compare other strategies against.

    Params:
        wait: The number of events to skip before emitting the first
            buy signals. This is useful when a strategy should not act on
            the very first events, for example while prices or other data
            are still warming up.
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
