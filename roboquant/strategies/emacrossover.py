from roboquant.signal import Signal
from roboquant.strategies.strategy import Strategy
from roboquant.event import Event


class EMACrossover(Strategy):
    """EMA Crossover Strategy"""

    def __init__(self, fast_period=13, slow_period=26, smoothing=2.0, price_type="DEFAULT"):
        super().__init__()
        self._history: dict[str, EMACrossover.__EMACalculator] = {}
        self.fast = 1.0 - (smoothing / (fast_period + 1))
        self.slow = 1.0 - (smoothing / (slow_period + 1))
        self.price_type = price_type
        self.min_steps = max(fast_period, slow_period)

    def create_signals(self, event: Event) -> dict[str, Signal]:
        signals: dict[str, Signal] = {}
        for symbol, item in event.price_items.items():

            price = item.price(self.price_type)
            calculator = self._history.get(symbol)

            if calculator is None:
                self._history[symbol] = self.__EMACalculator(self.fast, self.slow, price)
            elif not calculator.step >= self.min_steps:
                calculator.add_price(price)
            else:
                old_direction = calculator.get_direction()
                calculator.add_price(price)
                new_direction = calculator.get_direction()
                if old_direction != new_direction:
                    signals[symbol] = Signal(new_direction)

        return signals

    def reset(self):
        self._history = {}

    class __EMACalculator:

        __slots__ = "fast", "slow", "emaFast", "emaSlow", "step"

        def __init__(self, fast, slow, price):
            self.fast = fast
            self.slow = slow
            self.emaFast = price
            self.emaSlow = price
            self.step = 1

        def add_price(self, price: float):
            fast, slow = self.fast, self.slow
            self.emaFast = self.emaFast * fast + (1.0 - fast) * price
            self.emaSlow = self.emaSlow * slow + (1.0 - slow) * price
            self.step += 1

        def get_direction(self) -> float:
            return 1.0 if self.emaFast > self.emaSlow else -1.0
