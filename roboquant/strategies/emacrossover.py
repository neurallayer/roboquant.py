from roboquant.signal import Signal
from roboquant.asset import Asset
from roboquant.event import Event
from roboquant.strategies.strategy import Strategy


class EMACrossover(Strategy):
    """Trading strategy based on a fast and slow Exponential Moving Average.

    This strategy tracks each asset independently and emits a buy signal when
    the fast EMA crosses above the slow EMA, and a sell signal when it crosses
    below. Signals are only generated after enough prices have been observed to
    cover the longer EMA period.

    The constructor controls the fast and slow periods, smoothing factor,
    and the event price type used as input.
    """

    def __init__(self, fast_period: int = 13, slow_period: int = 26, smoothing: float = 2.0, price_type: str = "DEFAULT"):
        super().__init__()
        self._history: dict[Asset, EMACrossover._Calculator] = {}
        self.fast = 1.0 - (smoothing / (fast_period + 1))
        self.slow = 1.0 - (smoothing / (slow_period + 1))
        self.price_type = price_type
        self.min_steps = max(fast_period, slow_period)

    def create_signals(self, event: Event) -> list[Signal]:
        result : list[Signal] = []
        for asset, price in event.get_prices(self.price_type).items():
            if asset not in self._history:
                self._history[asset] = self._Calculator(self.fast, self.slow, price=price)
            else:
                calculator = self._history[asset]
                old_rating = calculator.is_above()
                step = calculator.add_price(price)

                if step > self.min_steps:
                    new_rating = calculator.is_above()
                    if old_rating != new_rating:
                        if new_rating:
                            result.append(Signal.buy(asset))
                        else:
                            result.append(Signal.sell(asset))
        return result

    class _Calculator:
        """Calculates the EMA crossover for a single asset"""

        __slots__ = "momentum1", "momentum2", "price1", "price2", "step"

        def __init__(self, momentum1: float, momentum2: float, price: float):
            self.momentum1 = momentum1
            self.momentum2 = momentum2
            self.price1 = price
            self.price2 = price
            self.step = 0

        def is_above(self) -> bool:
            """Return True is the first momentum is above the second momentum, False otherwise"""
            return self.price1 > self.price2

        def add_price(self, price: float) -> int:
            m1, m2 = self.momentum1, self.momentum2
            self.price1 = m1 * self.price1 + (1.0 - m1) * price
            self.price2 = m2 * self.price2 + (1.0 - m2) * price
            self.step += 1
            return self.step
