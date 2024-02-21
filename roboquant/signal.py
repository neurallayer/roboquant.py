from dataclasses import dataclass
from enum import Flag, auto


class SignalType(Flag):
    """Type of signal, either EXTRY, EXIT or BOTH"""
    ENTRY = auto()
    EXIT = auto()
    BOTH = ENTRY | EXIT


@dataclass(slots=True, frozen=True)
class Signal:
    """Signal that a strategy can create"""

    rating: float
    type: SignalType = SignalType.BOTH

    @staticmethod
    def BUY(signal_type=SignalType.BOTH):
        return Signal(1.0, signal_type)

    @staticmethod
    def SELL(signal_type=SignalType.BOTH):
        return Signal(-1.0, signal_type)

    def is_buy(self):
        return self.rating > 0.0

    def is_sell(self):
        return self.rating < 0.0


if __name__ == "__main__":
    s = Signal.BUY()
    print(s)
