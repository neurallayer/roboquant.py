from dataclasses import dataclass
from enum import Flag, auto


class SignalType(Flag):
    """Type of signal, either ENTRY, EXIT or BOTH"""

    ENTRY = auto()
    EXIT = auto()
    BOTH = ENTRY | EXIT


@dataclass(slots=True, frozen=True)
class Signal:
    """Signal that a strategy can create"""

    rating: float
    type: SignalType = SignalType.BOTH

    @staticmethod
    def buy(signal_type=SignalType.BOTH):
        """Create a BUY signal with rating of 1.0"""
        return Signal(1.0, signal_type)

    @staticmethod
    def sell(signal_type=SignalType.BOTH):
        """Create a SELL signal with rating of -1.0"""
        return Signal(-1.0, signal_type)

    @property
    def is_buy(self):
        return self.rating > 0.0

    @property
    def is_sell(self):
        return self.rating < 0.0

    @property
    def is_entry(self):
        return SignalType.ENTRY in self.type

    @property
    def is_exit(self):
        return SignalType.EXIT in self.type
