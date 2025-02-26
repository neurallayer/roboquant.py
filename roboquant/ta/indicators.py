import numpy as np

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

if __name__ == "__main__":
    data = np.random.rand(5, 100)
    close = data[3]
    print(sma(close, 10))
    print(atr(data[1], data[2], data[3]))
