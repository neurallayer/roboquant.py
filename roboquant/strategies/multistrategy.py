from itertools import groupby
from statistics import mean
from typing import Literal

from roboquant.event import Event
from roboquant.signal import Signal
from roboquant.strategies.strategy import Strategy


class MultiStrategy(Strategy):
    """Combine multiple strategies. The MultiStrategy provides additional control on how to handle conflicting
    signals for the same asset via the signal_filter:

    - first: in case of multiple signals for the same asset, the first one prevails.
    - last:  in case of multiple signals for the same asset, the last one prevails.
    - mean: return the mean of the signal ratings. All signals will be `ENTRY_EXIT`.
        If the mean is 0, no signal will be created for that asset.
    - none: return all signals and don't handle conflicts. This is also the default.
    """

    def __init__(
        self,
        *strategies: Strategy,
        signal_filter: Literal["last", "first", "none", "mean"] = "none"
    ):
        super().__init__()
        self.strategies = list(strategies)
        self.filter = signal_filter

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
