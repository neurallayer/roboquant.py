from typing import Protocol

from roboquant.event import Event


class Strategy(Protocol):
    """A strategy gives ratings based on incoming events and the items these events contain."""

    def give_ratings(self, event: Event) -> dict[str, float]:
        """Give a rating to zero or more symbols. Ratings are returned as a dictionary with key being the symbol name and
        the value being the rating.

        The rating is an float number with -1.0 being the strongest SELL rating an 1.0 being the strongest BUY rating.
        It is up to the used `Trader` to use this rating.

        For example, the default FlexTrader adjusts the order size based on the rating value. So a positive rating of 1.0
        creates a larger BUY order than a rating of 0.5
        """
        ...
