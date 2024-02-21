from typing import Literal
from roboquant.signal import Signal
from roboquant.strategies.strategy import Strategy
from roboquant.event import Event


class MultiStrategy(Strategy):
    """Combine one or more strategies. The MultiStrategy provides additional control on how to handle conflicting
    signals for the same symbols:

    - avg: average the signal ratings, and optionally apply weights based on the stratetegy that generated the rating
    - first: in case of multiple signals for a symbol, the first strategy wins
    - last:  in case of multiple signals for a symbol, the last strategy wins. This is also the default policy
    """

    def __init__(
        self, *strategies: Strategy, policy: Literal["last", "first"] = "last", weights: list[float] | None = None
    ):
        self.strategies = list(strategies)
        self.policy = policy
        self.weights = weights or [1.0] * len(strategies)
        assert len(self.weights) == len(self.strategies), "weights and strategies should be of equal lenght"

    def create_signals(self, event: Event) -> dict[str, Signal]:
        all_signals: list[dict[str, Signal]] = []
        for strategy in self.strategies:
            signals = strategy.create_signals(event)
            all_signals.append(signals)

        result = {}
        match self.policy:
            case "last":
                for signals in all_signals:
                    result.update(signals)
            case "first":
                for signals in reversed(all_signals):
                    result.update(signals)

        return result
