from itertools import groupby
from statistics import mean
from typing import Literal

from roboquant.event import Event
from roboquant.signal import Signal
from roboquant.strategies.strategy import Strategy


class MultiStrategy(Strategy):
    """Combine one or more signal strategies. The MultiStrategy provides additional control on how to handle conflicting
    signals for the same symbols via the signal_filter:

    - first: in case of multiple signals for the same symbol, the first one wins
    - last:  in case of multiple signals for the same symbol, the last one wins.
    - mean: return the mean of the signal ratings. All signals will be ENTRY and EXIT.
    - none: return all signals. This is also the default.
    """

    def __init__(
        self,
        *strategies: Strategy,
        order_filter: Literal["last", "first", "none", "mean"] = "none"
    ):
        super().__init__()
        self.strategies = list(strategies)
        self.filter = order_filter

    def create_signals(self, event: Event) -> list[Signal]:
        signals: list[Signal] = []
        for strategy in self.strategies:
            signals += strategy.create_signals(event)

        match self.filter:
            case "none":
                return signals
            case "last":
                s = {s.asset: s for s in signals}
                return list(s.values())
            case "first":
                s = {s.asset: s for s in reversed(signals)}
                return list(s.values())
            case "mean":
                result = []
                for key, group in groupby(signals, lambda signal: signal.asset):
                    rating = mean(signal.rating for signal in group)
                    if rating:
                        result.append(Signal(key, rating))
                return result

        raise ValueError("unsupported signal filter")
