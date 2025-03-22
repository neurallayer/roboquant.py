from dataclasses import dataclass
from enum import Flag, auto

from roboquant.asset import Asset


class SignalType(Flag):
    """Indicates how a signal can be used, either:

    - ENTRY: enter or increase a position size
    - EXIT: close or reduce a position size
    - ENTRY_EXIT: can be used both to enter/increase or close/reduce position sizes.
    """

    ENTRY = auto()
    """Indicates that this signal can be used to enter or increase a position size"""

    EXIT = auto()
    """Indicates that this signal can be used to exit or reduce a position size"""

    ENTRY_EXIT = ENTRY | EXIT
    """Indicates that this signal can be used for both entering and exiting a position size"""

    def __repr__(self):
        return self.name.split('.')[-1] # type: ignore


@dataclass(slots=True, frozen=True)
class Signal:
    """Signal that a strategy can create that indicates a BUY or SELL of an asset.
    It contains both the rating and the type of signal.

    A rating is a float normally between -1.0 and 1.0, where -1.0 is a strong sell, and 1.0 is a strong buy.
    But this range isn't enforced. It is up to the used `Trader` to use these values when converting signals to orders.

    The type indicates if it is an `ENTRY`, `EXIT` or `ENTRY_EXIT` signal. The default is `ENTRY_EXIT`. Please note that
    it is again up to the `trader` to handle these types correctly.

    Examples:
        ```python
        apple = Stock("AAPL")
        signal1 = Signal.buy(apple)
        signal2 = Signal.sell(apple, SignalType.EXIT)
        signal3 = Signal(apple, 0.5, SignalType.ENTRY)
        ```
    """

    asset: Asset
    """The asset this signal is for"""

    rating: float
    """The rating of this signal, normally a float value between -1.0 and 1.0"""

    type: SignalType = SignalType.ENTRY_EXIT
    """The type of signal, either ENTRY, EXIT or ENTRY_EXIT. Default is ENTRY_EXIT"""

    @staticmethod
    def buy(asset: Asset, signal_type: SignalType=SignalType.ENTRY_EXIT) -> "Signal":
        """Create a BUY signal with a rating of 1.0

        Args:
            asset (Asset): The asset this signal is for.
            signal_type (SignalType): The type of signal. Default is ENTRY_EXIT.

        Returns:
            Signal: A new Signal instance with a rating of 1.0.
        """
        return Signal(asset, 1.0, signal_type)

    @staticmethod
    def sell(asset: Asset, signal_type: SignalType=SignalType.ENTRY_EXIT) -> "Signal":
        """Create a SELL signal with a rating of -1.0

        Args:
            asset (Asset): The asset this signal is for.
            signal_type (SignalType): The type of signal. Default is ENTRY_EXIT.

        Returns:
            Signal: A new Signal instance with a rating of -1.0.
        """
        return Signal(asset, -1.0, signal_type)

    @property
    def is_buy(self) -> bool:
        """Return True if this is a BUY signal, False otherwise

        Returns:
            bool: True if this is a BUY signal, False otherwise.
        """
        return self.rating > 0.0

    @property
    def is_sell(self) -> bool:
        """Return True if this is a SELL signal, False otherwise

        Returns:
            bool: True if this is a SELL signal, False otherwise.
        """
        return self.rating < 0.0

    @property
    def is_entry(self) -> bool:
        """Return True if this is an ENTRY or ENTRY_EXIT signal, False otherwise

        Returns:
            bool: True if this is an ENTRY or ENTRY_EXIT signal, False otherwise.
        """
        return SignalType.ENTRY in self.type

    @property
    def is_exit(self) -> bool:
        """Return True if this is an EXIT or ENTRY_EXIT signal, False otherwise

        Returns:
            bool: True if this is an EXIT or ENTRY_EXIT signal, False otherwise.
        """
        return SignalType.EXIT in self.type
