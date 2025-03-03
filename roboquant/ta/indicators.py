"""series of technical indicators"""""


import numpy as np


EPS = 0.000000001


def _get_ema_weights(period, alpha):
    w = alpha * (1.0 - alpha) ** np.arange(period)
    w = np.flip(w) / np.sum(w)
    return w

def sma(data: np.ndarray, period: int):
    """
    Calculate the Simple Moving Average (SMA) of the given data over the specified period.

    Parameters:
    data (np.ndarray): Array of data points.
    period (int): Number of periods to calculate the SMA.

    Returns:
    float: The SMA of the data.
    """
    return float(np.mean(data[-period:]))


def median(data: np.ndarray, period: int):
    """
    Calculate the median of the last 'period' elements in the given data array.
    Parameters:
        data (np.ndarray): The input array containing numerical data.
        period (int): The number of elements from the end of the array to consider for the median calculation.
    Returns:
        float: The median value of the specified period of elements in the data array.
    """
    return float(np.median(data[-period:]))


def bb(data: np.ndarray, period: int, multiplier: float=2.0):
    """
    Calculate the Bollinger Bands of the given data over the specified period.

    Parameters:
    data (np.ndarray): Array of data points.
    period (int): Number of periods to calculate the Bollinger Bands.
    multiplier (float): Multiplier for the standard deviation.

    Returns:
    tuple: Upper band, basis (SMA), and lower band.
    """
    data = data[-period:]
    basis = float(np.mean(data))
    band = float(np.std(data)) * multiplier
    return basis + band, basis, basis - band


def rsi(data: np.ndarray, period: int):
    """
    Calculate the Relative Strength Index (RSI) of the given data over the specified period.

    Parameters:
    data (np.ndarray): Array of data points.
    period (int): Number of periods to calculate the RSI.

    Returns:
    float: The RSI of the data.
    """
    data = data[-period:]
    returns = np.diff(data)
    gain = np.mean(returns, where=returns > 0.0) or EPS
    loss = (- np.mean(returns, where=returns < 0.0)) or EPS
    rs = gain / loss
    return 100.0 - (100.0 / (1 + rs))


def atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14):
    """
    Calculate the Average True Range (ATR) of the given data over the specified period.

    Parameters:
    high (np.ndarray): Array of high prices.
    low (np.ndarray): Array of low prices.
    close (np.ndarray): Array of close prices.
    period (int): Number of periods to calculate the ATR.

    Returns:
    float: The ATR of the data.
    """
    h = high[-period:]
    l = low[-period:]  # noqa: E741
    c = close[-period-1:-1]

    high_low = np.abs(h - l)
    high_close = np.abs(h - c)
    low_close = np.abs(l - c)

    ranges = np.concat([high_low, high_close, low_close], axis=0)
    true_range = np.max(ranges, axis=0)
    return float(np.mean(true_range))


def ema(data: np.ndarray, period: int, alpha=0.1) -> float:
    """
    Calculate the Exponential Moving Average (EMA) of the given data over the specified period.

    Parameters:
    data (np.ndarray): Array of data points.
    period (int): Number of periods to calculate the EMA.
    alpha (float): Smoothing factor.

    Returns:
    float: The EMA of the data.
    """
    w = alpha * (1.0 - alpha) ** np.arange(period)
    w = np.flip(w) / np.sum(w)
    total = np.sum(w * data[-period:])
    return float(total)


def ema_crossover(data: np.ndarray, period1, period2, alpha=0.1):
    """
    Determine if an EMA crossover has occurred between two periods.

    Parameters:
    data (np.ndarray): Array of data points.
    period1 (int): First period for EMA calculation.
    period2 (int): Second period for EMA calculation.
    alpha (float): Smoothing factor.

    Returns:
    bool: True if a crossover has occurred, False otherwise.
    """
    new = np.sign(ema(data, period1, alpha) - ema(data, period2, alpha))
    old = np.sign(ema(data[:-1], period1, alpha) - ema(data[:-1], period2, alpha) )
    return new != old

def macd(data: np.ndarray, ema1 = 26, ema2 = 12, ema3 = 9):
    w1 = _get_ema_weights(ema1, 0.1)
    ema_1 = np.convolve(data[:-ema1-ema3], w1)

    w2 = _get_ema_weights(ema2, 0.1)
    ema_2 = np.convolve(data[:-ema2-ema3], w2)

    macd_line = ema_1 - ema_2
    macd_ema = ema(macd_line, ema3)
    return macd_ema > macd_line[-1]


def wma(data: np.ndarray, weights: np.ndarray):
    """
    Calculate the Weighted Moving Average (WMA) of the given data using the specified weights.

    Parameters:
    data (np.ndarray): Array of data points.
    weights (np.ndarray): Array of weights.

    Returns:
    float: The WMA of the data.
    """
    period = len(weights)
    weights = weights / np.sum(weights)
    return np.sum(data[-period:] * weights)


if __name__ == "__main__":
    data = np.random.rand(5, 1_000)
    close = data[3]


    print(sma(close, 10))
    print(ema(close, 14))

    for i in range(1,100):
        p = close[:-i]
        print(ema_crossover(p, 10, 14))


    print(atr(data[1], data[2], data[3]))

