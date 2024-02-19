from roboquant.strategies.strategy import Strategy
from roboquant.event import Event


class NOPStrategy(Strategy):
    """This strategy will never generate a rating.

    This is useful if you decide to implement all the trading logic in a Trader and don't require a Strategy.
    """

    def give_ratings(self, event: Event) -> dict[str, float]:
        """Always return an empty dict"""
        return {}
