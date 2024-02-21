from roboquant.signal import Signal
from roboquant.strategies.strategy import Strategy
from roboquant.event import Event


class NOPStrategy(Strategy):
    """This strategy will never generate a rating.

    This is useful if you decide to implement all the trading logic in a Trader and don't require a Strategy.
    """

    def create_signals(self, event: Event) -> dict[str, Signal]:
        """Always return an empty dict"""
        return {}
