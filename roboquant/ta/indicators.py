import numpy as np
import math

EPS = 0.000000001

def sma(data: np.ndarray, period: int):
    return float(np.mean(data[-period:]))


def bb(data: np.ndarray, period: int, multiplier: float=2.0):
    data = data[-period:]
    basis = float(np.mean(data))
    band = float(np.std(data)) * multiplier
    return basis + band, basis, basis - band


def rsi(data: np.ndarray, period: int):
    data = data[-period:]
    returns = np.diff(data)
    gain = np.mean(returns, where=returns > 0.0) or EPS
    loss = (- np.mean(returns, where=returns < 0.0)) or EPS
    rs = gain / loss
    return 100.0 - (100.0 / (1 + rs))

def atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14):
    h = high[-period:]
    l = low[-period:]  # noqa: E741
    c = close[-period-1:-1]

    high_low = np.abs(h - l)
    high_close = np.abs(h - c)
    low_close = np.abs(l - c)

    ranges = np.concat([high_low, high_close, low_close], axis=0)
    true_range = np.max(ranges, axis=0)
    return float(np.mean(true_range))


class SMA:

    def __init__(self, period: int) -> None:
        self.period = period

    def __call__(self, data: np.ndarray):
        return np.mean(data[-self.period:])


class WeightedMA:
    """Use a Weighted Moving Average. The length of the weights is the minimum required period.
    It is ensured that the weights add up to 1.0.
    """

    def __init__(self, weights: np.ndarray) -> None:
            # Make sure the weights add up to 1
            self._w = np.array(weights) / np.sum(weights)

            # Weights should add up to 1
            assert math.isclose(np.sum(self._w), 1.0)

    def __call__(self, data: np.ndarray):
        period = len(self._w)
        return np.sum(self._w * data[-period:])


class EMA(WeightedMA):

    def __init__(self, period: int, alpha=0.1) -> None:
        # Calculate the weights
        w = (1.0 - alpha) ** np.arange(period)
        w = np.flip(w)
        super().__init__(w)


def ema2(data: np.ndarray, period: int, alpha=0.1):
    data = data[-period:]
    w = np.flip(alpha * (1.0 - alpha) ** np.arange(period))
    total = np.sum(w * data)
    return total


if __name__ == "__main__":
    data = np.random.rand(5, 1_000)
    close = data[3]

    ema_10 = EMA(10, 0.1)
    sma_10 = SMA(10)

    print(sma_10(close))
    print(ema_10(close))
    print(atr(data[1], data[2], data[3]))

