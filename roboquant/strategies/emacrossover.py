from roboquant.event import Event
from roboquant.signal import Signal
from roboquant.strategies.strategy import Strategy


class EMACrossover(Strategy):
    """EMA Crossover Strategy"""

    def __init__(self, fast_period=13, slow_period=26, smoothing=2.0, price_type="DEFAULT"):
        super().__init__()
        self._history = {}
        self.fast = 1.0 - (smoothing / (fast_period + 1))
        self.slow = 1.0 - (smoothing / (slow_period + 1))
        self.price_type = price_type
        self.step = 0
        self.min_steps = max(fast_period, slow_period)

    def create_signals(self, event: Event) -> dict[str, Signal]:
        signals: dict[str, Signal] = {}
        for symbol, item in event.price_items.items():

            price = item.price(self.price_type)
            calculators = self._history.get(symbol)

            if calculators is None:
                self._history[symbol] = self.__EMACalculator(self.fast, price), self.__EMACalculator(self.slow, price)
            else:
                old_rating = calculators[0].price > calculators[1].price
                calculators[0].add_price(price)
                calculators[1].add_price(price)

                if self.step > self.min_steps:
                    new_rating = calculators[0].price > calculators[1].price
                    if old_rating != new_rating:
                        signals[symbol] = Signal.buy() if new_rating else Signal.sell()

        self.step += 1
        return signals

    class __EMACalculator:

        __slots__ = "momentum", "price", "step"

        def __init__(self, momentum, price):
            self.momentum = momentum
            self.price = price

        def add_price(self, price: float):
            self.price = self.momentum * self.price + (1.0 - self.momentum) * price
