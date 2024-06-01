from typing import Literal
from itertools import groupby
from statistics import mean

from roboquant.event import Event
from roboquant.strategies.signal import Signal
from roboquant.strategies.signalstrategy import SignalStrategy


class MultiStrategy(SignalStrategy):
    """Combine one or more signal strategies. The MultiStrategy provides additional control on how to handle conflicting
    signals for the same symbols via the signal_filter:

    - first: in case of multiple signals for the same symbol, the first one wins
    - last:  in case of multiple signals for the same symbol, the last one wins.
    - avg: return the avgerage of the signals. All signals will be ENTRY and EXIT.
    - none: return all signals. This is also the default.
    """

    def __init__(self, *strategies: SignalStrategy, signal_filter: Literal["last", "first", "avg", "none"] = "none"):
        super().__init__()
        self.strategies = list(strategies)
        self.signal_filter = signal_filter

    def create_signals(self, event: Event):
        signals: list[Signal] = []
        for strategy in self.strategies:
            signals += strategy.create_signals(event)

        match self.signal_filter:
            case "none":
                return signals
            case "last":
                s = {s.symbol: s for s in signals}
                return list(s.values())
            case "first":
                s = {s.symbol: s for s in reversed(signals)}
                return list(s.values())
            case "avg":
                result = []
                g = groupby(signals, lambda x: x.symbol)
                for symbol, v in g:
                    rating = mean(s.rating for s in v)
                    result.append(Signal(symbol, rating))
                return result

        raise ValueError("unsupported signal filter")
