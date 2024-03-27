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
    """Signal that a strategy can create.It contains both a rating and the type of signal.

    A rating is a float normally between -1.0 and 1.0, where -1.0 is a strong sell and 1.0 is a strong buy.
    But in cases it can exceed these values. It is up to the used trader to handle these values

    Examples:
    ```
    Signal.buy("XYZ")
    Signal.sell("XYZ", SignalType.EXIT)
    Signal("XYZ", 0.5, SignalType.ENTRY)
    ```
    """
    symbol: str
    rating: float
    type: SignalType = SignalType.ENTRY_EXIT

    @staticmethod
    def buy(symbol, signal_type=SignalType.ENTRY_EXIT):
        """Create a BUY signal with a rating of 1.0"""
        return Signal(symbol, 1.0, signal_type)

    @staticmethod
    def sell(symbol, signal_type=SignalType.ENTRY_EXIT):
        """Create a SELL signal with a rating of -1.0"""
        return Signal(symbol, -1.0, signal_type)

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
