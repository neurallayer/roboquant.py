from dataclasses import dataclass
from enum import Flag, auto


class SignalType(Flag):
    """Type of signal, either ENTRY, EXIT or ENTRY_EXIT"""

    ENTRY = auto()
    EXIT = auto()
    ENTRY_EXIT = ENTRY | EXIT

    def __str__(self):
        return self.name


@dataclass(slots=True, frozen=True)
class Signal:
    """Signal that a strategy can create.
    It contains both a rating between -1.0 and 1.0 and the type of signal.

    Examples:
    ```
    Signal.buy("XYZ")
    Signal.sell("XYZ", SignalType.EXIT)
    Signal("XYZ", 0.5, SignalType.ENTRY)
    ```
    """

    rating: float
    type: SignalType = SignalType.ENTRY_EXIT

    @staticmethod
    def buy(signal_type=SignalType.ENTRY_EXIT):
        """Create a BUY signal with a rating of 1.0"""
        return Signal(1.0, signal_type)

    @staticmethod
    def sell(signal_type=SignalType.ENTRY_EXIT):
        """Create a SELL signal with a rating of -1.0"""
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


BUY = Signal.buy(SignalType.ENTRY_EXIT)
"""BUY signal with a rating of 1.0 and valid for both entry and exit signals"""

SELL = Signal.sell(SignalType.ENTRY_EXIT)
"""SELL signal with a rating of -1.0 and valid for both entry and exit signals"""
