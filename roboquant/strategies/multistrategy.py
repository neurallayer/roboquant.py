from typing import Literal

from roboquant.event import Event
from roboquant.strategies.strategy import Strategy


class MultiStrategy(Strategy):
    """Combine one or more strategies. The MultiStrategy provides additional control on how to handle conflicting
    signals for the same symbols:

    - first: in case of multiple signals for a symbol, the first strategy wins
    - last:  in case of multiple signals for a symbol, the last strategy wins. This is also the default policy
    """

    def __init__(self, *strategies: Strategy, policy: Literal["last", "first", "all"] = "last"):
        self.strategies = list(strategies)
        self.policy = policy

    def create_signals(self, event: Event):
        signals = []
        for strategy in self.strategies:
            tmp = strategy.create_signals(event)
            signals += tmp

        match self.policy:
            case "last":
                s = {s.symbol: s for s in signals}
                return list(s.values())
            case "first":
                s = {s.symbol: s for s in reversed(signals)}
                return list(s.values())
            case "all":
                return signals
