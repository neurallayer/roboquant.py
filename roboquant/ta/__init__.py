"""Set of stubs for the streaming version of the ta-lib indicators that makes them discoverable and typed."""

import numpy as np
import logging
from enum import Enum
from typing import Tuple
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

try:
    import talib._ta_lib as _ta_lib  # type: ignore
except ImportError:
    logger.warning("TA-Lib is not installed, TA functions will raise an exception")


class MA_Type(Enum):
    SMA = 0
    EMA = 1
    WMA = 2
    DEMA = 3
    TEMA = 4
    TRIMA = 5
    KAMA = 6
    MAMA = 7
    T3 = 8


_exception = NotImplementedError("Not implemented, is TA-LIB installed?")

# Overlap Studies Functions


def BBANDS(
    real: NDArray[np.float64], timeperiod: int = 5, nbdevup: float = 2, nbdevdn: float = 2, matype: MA_Type = MA_Type.SMA
) -> Tuple[float, float, float]:
    raise _exception


def DEMA(real: NDArray[np.float64], timeperiod: int = 30) -> float:
    raise _exception


def EMA(real: NDArray[np.float64], timeperiod: int = 30) -> float:
    raise _exception


def HT_TRENDLINE(real: NDArray[np.float64]) -> float:
    raise _exception


def KAMA(real: NDArray[np.float64], timeperiod: int = 30) -> float:
    raise _exception


def MA(real: NDArray[np.float64], timeperiod: int = 30, matype: MA_Type = MA_Type.SMA) -> float:
    raise _exception


def MAMA(real: NDArray[np.float64], fastlimit: float = 0, slowlimit: float = 0) -> Tuple[float, float]:
    raise _exception


def MAVP(
    real: NDArray[np.float64], periods: float, minperiod: int = 2, maxperiod: int = 30, matype: MA_Type = MA_Type.SMA
) -> float:
    raise _exception


def MIDPOINT(real: NDArray[np.float64], timeperiod: int = 14) -> float:
    raise _exception


def MIDPRICE(high: NDArray[np.float64], low: NDArray[np.float64], timeperiod: int = 14) -> float:
    raise _exception


def SAR(high: NDArray[np.float64], low: NDArray[np.float64], acceleration: float = 0, maximum: float = 0) -> float:
    raise _exception


def SAREXT(
    high: NDArray[np.float64],
    low: NDArray[np.float64],
    startvalue: float = 0,
    offsetonreverse: float = 0,
    accelerationinitlong: float = 0,
    accelerationlong: float = 0,
    accelerationmaxlong: float = 0,
    accelerationinitshort: float = 0,
    accelerationshort: float = 0,
    accelerationmaxshort: float = 0,
) -> float:
    raise _exception


def SMA(real: NDArray[np.float64], timeperiod: int = 30) -> float:
    raise _exception


def T3(real: NDArray[np.float64], timeperiod: int = 5, vfactor: float = 0) -> float:
    raise _exception


def TEMA(real: NDArray[np.float64], timeperiod: int = 30) -> float:
    raise _exception


def TRIMA(real: NDArray[np.float64], timeperiod: int = 30) -> float:
    raise _exception


def WMA(real: NDArray[np.float64], timeperiod: int = 30) -> float:
    raise _exception


# Momentum Indicator Functions


def ADX(high: NDArray[np.float64], low: NDArray[np.float64], close: NDArray[np.float64], timeperiod: int = 14) -> float:
    raise _exception


def ADXR(high: NDArray[np.float64], low: NDArray[np.float64], close: NDArray[np.float64], timeperiod: int = 14) -> float:
    raise _exception


def APO(real: NDArray[np.float64], fastperiod: int = 12, slowperiod: int = 26, matype: MA_Type = MA_Type.SMA) -> float:
    raise _exception


def AROON(high: NDArray[np.float64], low: NDArray[np.float64], timeperiod: int = 14) -> Tuple[float, float]:
    raise _exception


def AROONOSC(high: NDArray[np.float64], low: NDArray[np.float64], timeperiod: int = 14) -> float:
    raise _exception


def BOP(open: NDArray[np.float64], high: NDArray[np.float64], low: NDArray[np.float64], close: NDArray[np.float64]) -> float:
    raise _exception


def CCI(high: NDArray[np.float64], low: NDArray[np.float64], close: NDArray[np.float64], timeperiod: int = 14) -> float:
    raise _exception


def CMO(real: NDArray[np.float64], timeperiod: int = 14) -> float:
    raise _exception


def DX(high: NDArray[np.float64], low: NDArray[np.float64], close: NDArray[np.float64], timeperiod: int = 14) -> float:
    raise _exception


def MACD(
    real: NDArray[np.float64], fastperiod: int = 12, slowperiod: int = 26, signalperiod: int = 9
) -> Tuple[float, float, float]:
    raise _exception


def MACDEXT(
    real: NDArray[np.float64],
    fastperiod: int = 12,
    fastmatype: MA_Type = MA_Type.SMA,
    slowperiod: int = 26,
    slowmatype: MA_Type = MA_Type.SMA,
    signalperiod: int = 9,
    signalmatype: MA_Type = MA_Type.SMA,
) -> Tuple[float, float, float]:
    raise _exception


def MACDFIX(real: NDArray[np.float64], signalperiod: int = 9) -> Tuple[float, float, float]:
    raise _exception


def MFI(
    high: NDArray[np.float64],
    low: NDArray[np.float64],
    close: NDArray[np.float64],
    volume: NDArray[np.float64],
    timeperiod: int = 14,
) -> float:
    raise _exception


def MINUS_DI(high: NDArray[np.float64], low: NDArray[np.float64], close: NDArray[np.float64], timeperiod: int = 14) -> float:
    raise _exception


def MINUS_DM(high: NDArray[np.float64], low: NDArray[np.float64], timeperiod: int = 14) -> float:
    raise _exception


def MOM(real: NDArray[np.float64], timeperiod: int = 10) -> float:
    raise _exception


def PLUS_DI(high: NDArray[np.float64], low: NDArray[np.float64], close: NDArray[np.float64], timeperiod: int = 14) -> float:
    raise _exception


def PLUS_DM(high: NDArray[np.float64], low: NDArray[np.float64], timeperiod: int = 14) -> float:
    raise _exception


def PPO(real: NDArray[np.float64], fastperiod: int = 12, slowperiod: int = 26, matype: MA_Type = MA_Type.SMA) -> float:
    raise _exception


def ROC(real: NDArray[np.float64], timeperiod: int = 10) -> float:
    raise _exception


def ROCP(real: NDArray[np.float64], timeperiod: int = 10) -> float:
    raise _exception


def ROCR(real: NDArray[np.float64], timeperiod: int = 10) -> float:
    raise _exception


def ROCR100(real: NDArray[np.float64], timeperiod: int = 10) -> float:
    raise _exception


def RSI(real: NDArray[np.float64], timeperiod: int = 14) -> float:
    raise _exception


def STOCH(
    high: NDArray[np.float64],
    low: NDArray[np.float64],
    close: NDArray[np.float64],
    fastk_period: int = 5,
    slowk_period: int = 3,
    slowk_matype: MA_Type = MA_Type.SMA,
    slowd_period: int = 3,
    slowd_matype: MA_Type = MA_Type.SMA,
) -> Tuple[float, float]:
    raise _exception


def STOCHF(
    high: NDArray[np.float64],
    low: NDArray[np.float64],
    close: NDArray[np.float64],
    fastk_period: int = 5,
    fastd_period: int = 3,
    fastd_matype: MA_Type = MA_Type.SMA,
) -> Tuple[float, float]:
    raise _exception


def STOCHRSI(
    real: NDArray[np.float64],
    timeperiod: int = 14,
    fastk_period: int = 5,
    fastd_period: int = 3,
    fastd_matype: MA_Type = MA_Type.SMA,
) -> Tuple[float, float]:
    raise _exception


def TRIX(real: NDArray[np.float64], timeperiod: int = 30) -> float:
    raise _exception


def ULTOSC(
    high: NDArray[np.float64],
    low: NDArray[np.float64],
    close: NDArray[np.float64],
    timeperiod1: int = 7,
    timeperiod2: int = 14,
    timeperiod3: int = 28,
) -> float:
    raise _exception


def WILLR(high: NDArray[np.float64], low: NDArray[np.float64], close: NDArray[np.float64], timeperiod: int = 14) -> float:
    raise _exception


# Volume Indicator Functions


def AD(high: NDArray[np.float64], low: NDArray[np.float64], close: NDArray[np.float64], volume: NDArray[np.float64]) -> float:
    raise _exception


def ADOSC(
    high: NDArray[np.float64],
    low: NDArray[np.float64],
    close: NDArray[np.float64],
    volume: NDArray[np.float64],
    fastperiod: int = 3,
    slowperiod: int = 10,
) -> float:
    raise _exception


def OBV(close: NDArray[np.float64], volume: NDArray[np.float64]) -> float:
    raise _exception


# Volatility Indicator Functions


def ATR(high: NDArray[np.float64], low: NDArray[np.float64], close: NDArray[np.float64], timeperiod: int = 14) -> float:
    raise _exception


def NATR(high: NDArray[np.float64], low: NDArray[np.float64], close: NDArray[np.float64], timeperiod: int = 14) -> float:
    raise _exception


def TRANGE(high: NDArray[np.float64], low: NDArray[np.float64], close: NDArray[np.float64]) -> float:
    raise _exception


# Price Transform Functions


def AVGPRICE(
    open: NDArray[np.float64], high: NDArray[np.float64], low: NDArray[np.float64], close: NDArray[np.float64]
) -> float:
    raise _exception


def MEDPRICE(high: NDArray[np.float64], low: NDArray[np.float64]) -> float:
    raise _exception


def TYPPRICE(high: NDArray[np.float64], low: NDArray[np.float64], close: NDArray[np.float64]) -> float:
    raise _exception


def WCLPRICE(high: NDArray[np.float64], low: NDArray[np.float64], close: NDArray[np.float64]) -> float:
    raise _exception


# Cycle Indicator Functions


def HT_DCPERIOD(real: NDArray[np.float64]) -> float:
    raise _exception


def HT_DCPHASE(real: NDArray[np.float64]) -> float:
    raise _exception


def HT_PHASOR(real: NDArray[np.float64]) -> Tuple[float, float]:
    raise _exception


def HT_SINE(real: NDArray[np.float64]) -> Tuple[float, float]:
    raise _exception


def HT_TRENDMODE(real: NDArray[np.float64]) -> Tuple[float, int]:
    raise _exception


# Pattern Recognition Functions


def CDL2CROWS(
    open: NDArray[np.float64], high: NDArray[np.float64], low: NDArray[np.float64], close: NDArray[np.float64]
) -> int:
    raise _exception


def CDL3BLACKCROWS(
    open: NDArray[np.float64], high: NDArray[np.float64], low: NDArray[np.float64], close: NDArray[np.float64]
) -> int:
    raise _exception


def CDL3INSIDE(
    open: NDArray[np.float64], high: NDArray[np.float64], low: NDArray[np.float64], close: NDArray[np.float64]
) -> int:
    raise _exception


def CDL3LINESTRIKE(
    open: NDArray[np.float64], high: NDArray[np.float64], low: NDArray[np.float64], close: NDArray[np.float64]
) -> int:
    raise _exception


def CDL3OUTSIDE(
    open: NDArray[np.float64], high: NDArray[np.float64], low: NDArray[np.float64], close: NDArray[np.float64]
) -> int:
    raise _exception


def CDL3STARSINSOUTH(
    open: NDArray[np.float64], high: NDArray[np.float64], low: NDArray[np.float64], close: NDArray[np.float64]
) -> int:
    raise _exception


def CDL3WHITESOLDIERS(
    open: NDArray[np.float64], high: NDArray[np.float64], low: NDArray[np.float64], close: NDArray[np.float64]
) -> int:
    raise _exception


def CDLABANDONEDBABY(
    open: NDArray[np.float64],
    high: NDArray[np.float64],
    low: NDArray[np.float64],
    close: NDArray[np.float64],
    penetration: float = 0,
) -> int:
    raise _exception


def CDLADVANCEBLOCK(
    open: NDArray[np.float64], high: NDArray[np.float64], low: NDArray[np.float64], close: NDArray[np.float64]
) -> int:
    raise _exception


def CDLBELTHOLD(
    open: NDArray[np.float64], high: NDArray[np.float64], low: NDArray[np.float64], close: NDArray[np.float64]
) -> int:
    raise _exception


def CDLBREAKAWAY(
    open: NDArray[np.float64], high: NDArray[np.float64], low: NDArray[np.float64], close: NDArray[np.float64]
) -> int:
    raise _exception


def CDLCLOSINGMARUBOZU(
    open: NDArray[np.float64], high: NDArray[np.float64], low: NDArray[np.float64], close: NDArray[np.float64]
) -> int:
    raise _exception


def CDLCONCEALBABYSWALL(
    open: NDArray[np.float64], high: NDArray[np.float64], low: NDArray[np.float64], close: NDArray[np.float64]
) -> int:
    raise _exception


def CDLCOUNTERATTACK(
    open: NDArray[np.float64], high: NDArray[np.float64], low: NDArray[np.float64], close: NDArray[np.float64]
) -> int:
    raise _exception


def CDLDARKCLOUDCOVER(
    open: NDArray[np.float64],
    high: NDArray[np.float64],
    low: NDArray[np.float64],
    close: NDArray[np.float64],
    penetration: float = 0,
) -> int:
    raise _exception


def CDLDOJI(open: NDArray[np.float64], high: NDArray[np.float64], low: NDArray[np.float64], close: NDArray[np.float64]) -> int:
    raise _exception


def CDLDOJISTAR(
    open: NDArray[np.float64], high: NDArray[np.float64], low: NDArray[np.float64], close: NDArray[np.float64]
) -> int:
    raise _exception


def CDLDRAGONFLYDOJI(
    open: NDArray[np.float64], high: NDArray[np.float64], low: NDArray[np.float64], close: NDArray[np.float64]
) -> int:
    raise _exception


def CDLENGULFING(
    open: NDArray[np.float64], high: NDArray[np.float64], low: NDArray[np.float64], close: NDArray[np.float64]
) -> int:
    raise _exception


def CDLEVENINGDOJISTAR(
    open: NDArray[np.float64],
    high: NDArray[np.float64],
    low: NDArray[np.float64],
    close: NDArray[np.float64],
    penetration: float = 0,
) -> int:
    raise _exception


def CDLEVENINGSTAR(
    open: NDArray[np.float64],
    high: NDArray[np.float64],
    low: NDArray[np.float64],
    close: NDArray[np.float64],
    penetration: float = 0,
) -> int:
    raise _exception


def CDLGAPSIDESIDEWHITE(
    open: NDArray[np.float64], high: NDArray[np.float64], low: NDArray[np.float64], close: NDArray[np.float64]
) -> int:
    raise _exception


def CDLGRAVESTONEDOJI(
    open: NDArray[np.float64], high: NDArray[np.float64], low: NDArray[np.float64], close: NDArray[np.float64]
) -> int:
    raise _exception


def CDLHAMMER(
    open: NDArray[np.float64], high: NDArray[np.float64], low: NDArray[np.float64], close: NDArray[np.float64]
) -> int:
    raise _exception


def CDLHANGINGMAN(
    open: NDArray[np.float64], high: NDArray[np.float64], low: NDArray[np.float64], close: NDArray[np.float64]
) -> int:
    raise _exception


def CDLHARAMI(
    open: NDArray[np.float64], high: NDArray[np.float64], low: NDArray[np.float64], close: NDArray[np.float64]
) -> int:
    raise _exception


def CDLHARAMICROSS(
    open: NDArray[np.float64], high: NDArray[np.float64], low: NDArray[np.float64], close: NDArray[np.float64]
) -> int:
    raise _exception


def CDLHIGHWAVE(
    open: NDArray[np.float64], high: NDArray[np.float64], low: NDArray[np.float64], close: NDArray[np.float64]
) -> int:
    raise _exception


def CDLHIKKAKE(
    open: NDArray[np.float64], high: NDArray[np.float64], low: NDArray[np.float64], close: NDArray[np.float64]
) -> int:
    raise _exception


def CDLHIKKAKEMOD(
    open: NDArray[np.float64], high: NDArray[np.float64], low: NDArray[np.float64], close: NDArray[np.float64]
) -> int:
    raise _exception


def CDLHOMINGPIGEON(
    open: NDArray[np.float64], high: NDArray[np.float64], low: NDArray[np.float64], close: NDArray[np.float64]
) -> int:
    raise _exception


def CDLIDENTICAL3CROWS(
    open: NDArray[np.float64], high: NDArray[np.float64], low: NDArray[np.float64], close: NDArray[np.float64]
) -> int:
    raise _exception


def CDLINNECK(
    open: NDArray[np.float64], high: NDArray[np.float64], low: NDArray[np.float64], close: NDArray[np.float64]
) -> int:
    raise _exception


def CDLINVERTEDHAMMER(
    open: NDArray[np.float64], high: NDArray[np.float64], low: NDArray[np.float64], close: NDArray[np.float64]
) -> int:
    raise _exception


def CDLKICKING(
    open: NDArray[np.float64], high: NDArray[np.float64], low: NDArray[np.float64], close: NDArray[np.float64]
) -> int:
    raise _exception


def CDLKICKINGBYLENGTH(
    open: NDArray[np.float64], high: NDArray[np.float64], low: NDArray[np.float64], close: NDArray[np.float64]
) -> int:
    raise _exception


def CDLLADDERBOTTOM(
    open: NDArray[np.float64], high: NDArray[np.float64], low: NDArray[np.float64], close: NDArray[np.float64]
) -> int:
    raise _exception


def CDLLONGLEGGEDDOJI(
    open: NDArray[np.float64], high: NDArray[np.float64], low: NDArray[np.float64], close: NDArray[np.float64]
) -> int:
    raise _exception


def CDLLONGLINE(
    open: NDArray[np.float64], high: NDArray[np.float64], low: NDArray[np.float64], close: NDArray[np.float64]
) -> int:
    raise _exception


def CDLMARUBOZU(
    open: NDArray[np.float64], high: NDArray[np.float64], low: NDArray[np.float64], close: NDArray[np.float64]
) -> int:
    raise _exception


def CDLMATCHINGLOW(
    open: NDArray[np.float64], high: NDArray[np.float64], low: NDArray[np.float64], close: NDArray[np.float64]
) -> int:
    raise _exception


def CDLMATHOLD(
    open: NDArray[np.float64],
    high: NDArray[np.float64],
    low: NDArray[np.float64],
    close: NDArray[np.float64],
    penetration: float = 0,
) -> int:
    raise _exception


def CDLMORNINGDOJISTAR(
    open: NDArray[np.float64],
    high: NDArray[np.float64],
    low: NDArray[np.float64],
    close: NDArray[np.float64],
    penetration: float = 0,
) -> int:
    raise _exception


def CDLMORNINGSTAR(
    open: NDArray[np.float64],
    high: NDArray[np.float64],
    low: NDArray[np.float64],
    close: NDArray[np.float64],
    penetration: float = 0,
) -> int:
    raise _exception


def CDLONNECK(
    open: NDArray[np.float64], high: NDArray[np.float64], low: NDArray[np.float64], close: NDArray[np.float64]
) -> int:
    raise _exception


def CDLPIERCING(
    open: NDArray[np.float64], high: NDArray[np.float64], low: NDArray[np.float64], close: NDArray[np.float64]
) -> int:
    raise _exception


def CDLRICKSHAWMAN(
    open: NDArray[np.float64], high: NDArray[np.float64], low: NDArray[np.float64], close: NDArray[np.float64]
) -> int:
    raise _exception


def CDLRISEFALL3METHODS(
    open: NDArray[np.float64], high: NDArray[np.float64], low: NDArray[np.float64], close: NDArray[np.float64]
) -> int:
    raise _exception


def CDLSEPARATINGLINES(
    open: NDArray[np.float64], high: NDArray[np.float64], low: NDArray[np.float64], close: NDArray[np.float64]
) -> int:
    raise _exception


def CDLSHOOTINGSTAR(
    open: NDArray[np.float64], high: NDArray[np.float64], low: NDArray[np.float64], close: NDArray[np.float64]
) -> int:
    raise _exception


def CDLSHORTLINE(
    open: NDArray[np.float64], high: NDArray[np.float64], low: NDArray[np.float64], close: NDArray[np.float64]
) -> int:
    raise _exception


def CDLSPINNINGTOP(
    open: NDArray[np.float64], high: NDArray[np.float64], low: NDArray[np.float64], close: NDArray[np.float64]
) -> int:
    raise _exception


def CDLSTALLEDPATTERN(
    open: NDArray[np.float64], high: NDArray[np.float64], low: NDArray[np.float64], close: NDArray[np.float64]
) -> int:
    raise _exception


def CDLSTICKSANDWICH(
    open: NDArray[np.float64], high: NDArray[np.float64], low: NDArray[np.float64], close: NDArray[np.float64]
) -> int:
    raise _exception


def CDLTAKURI(
    open: NDArray[np.float64], high: NDArray[np.float64], low: NDArray[np.float64], close: NDArray[np.float64]
) -> int:
    raise _exception


def CDLTASUKIGAP(
    open: NDArray[np.float64], high: NDArray[np.float64], low: NDArray[np.float64], close: NDArray[np.float64]
) -> int:
    raise _exception


def CDLTHRUSTING(
    open: NDArray[np.float64], high: NDArray[np.float64], low: NDArray[np.float64], close: NDArray[np.float64]
) -> int:
    raise _exception


def CDLTRISTAR(
    open: NDArray[np.float64], high: NDArray[np.float64], low: NDArray[np.float64], close: NDArray[np.float64]
) -> int:
    raise _exception


def CDLUNIQUE3RIVER(
    open: NDArray[np.float64], high: NDArray[np.float64], low: NDArray[np.float64], close: NDArray[np.float64]
) -> int:
    raise _exception


def CDLUPSIDEGAP2CROWS(
    open: NDArray[np.float64], high: NDArray[np.float64], low: NDArray[np.float64], close: NDArray[np.float64]
) -> int:
    raise _exception


def CDLXSIDEGAP3METHODS(
    open: NDArray[np.float64], high: NDArray[np.float64], low: NDArray[np.float64], close: NDArray[np.float64]
) -> int:
    raise _exception


# Statistic Functions


def BETA(real0: NDArray[np.float64], real1: NDArray[np.float64], timeperiod: int = 5) -> float:
    raise _exception


def CORREL(real0: NDArray[np.float64], real1: NDArray[np.float64], timeperiod: int = 30) -> float:
    raise _exception


def LINEARREG(real: NDArray[np.float64], timeperiod: int = 14) -> float:
    raise _exception


def LINEARREG_ANGLE(real: NDArray[np.float64], timeperiod: int = 14) -> float:
    raise _exception


def LINEARREG_INTERCEPT(real: NDArray[np.float64], timeperiod: int = 14) -> float:
    raise _exception


def LINEARREG_SLOPE(real: NDArray[np.float64], timeperiod: int = 14) -> float:
    raise _exception


def STDDEV(real: NDArray[np.float64], timeperiod: int = 5, nbdev: float = 1) -> float:
    raise _exception


def TSF(real: NDArray[np.float64], timeperiod: int = 14) -> float:
    raise _exception


def VAR(real: NDArray[np.float64], timeperiod: int = 5, nbdev: float = 1) -> float:
    raise _exception


# Math Transform Functions


def ACOS(real: NDArray[np.float64]) -> float:
    raise _exception


def ASIN(real: NDArray[np.float64]) -> float:
    raise _exception


def ATAN(real: NDArray[np.float64]) -> float:
    raise _exception


def CEIL(real: NDArray[np.float64]) -> float:
    raise _exception


def COS(real: NDArray[np.float64]) -> float:
    raise _exception


def COSH(real: NDArray[np.float64]) -> float:
    raise _exception


def EXP(real: NDArray[np.float64]) -> float:
    raise _exception


def FLOOR(real: NDArray[np.float64]) -> float:
    raise _exception


def LN(real: NDArray[np.float64]) -> float:
    raise _exception


def LOG10(real: NDArray[np.float64]) -> float:
    raise _exception


def SIN(real: NDArray[np.float64]) -> float:
    raise _exception


def SINH(real: NDArray[np.float64]) -> float:
    raise _exception


def SQRT(real: NDArray[np.float64]) -> float:
    raise _exception


def TAN(real: NDArray[np.float64]) -> float:
    raise _exception


def TANH(real: NDArray[np.float64]) -> float:
    raise _exception


# Math Operator Functions


def ADD(real0: NDArray[np.float64], real1: NDArray[np.float64]) -> float:
    raise _exception


def DIV(real0: NDArray[np.float64], real1: NDArray[np.float64]) -> float:
    raise _exception


def MAX(real: NDArray[np.float64], timeperiod: int = 30) -> float:
    raise _exception


def MAXINDEX(real: NDArray[np.float64], timeperiod: int = 30) -> int:
    raise _exception


def MIN(real: NDArray[np.float64], timeperiod: int = 30) -> float:
    raise _exception


def MININDEX(real: NDArray[np.float64], timeperiod: int = 30) -> int:
    raise _exception


def MINMAX(real: NDArray[np.float64], timeperiod: int = 30) -> Tuple[float, float]:
    raise _exception


def MINMAXINDEX(real: NDArray[np.float64], timeperiod: int = 30) -> Tuple[float, float]:
    raise _exception


def MULT(real0: NDArray[np.float64], real1: NDArray[np.float64]) -> float:
    raise _exception


def SUB(real0: NDArray[np.float64], real1: NDArray[np.float64]) -> float:
    raise _exception


def SUM(real: NDArray[np.float64], timeperiod: int = 30) -> float:
    raise _exception


# ################
# missed ones
# ###############


def ACCBANDS(
    open: NDArray[np.float64], high: NDArray[np.float64], low: NDArray[np.float64], timeperiod: int = 20
) -> Tuple[float, float, float]:
    raise _exception


def AVGDEV(real: NDArray[np.float64], timeperiod: int = 14) -> float:
    raise _exception


def IMI(open: NDArray[np.float64], close: NDArray[np.float64], timeperiod: int = 14) -> float:
    raise _exception


# for fn in dict(globals()):
#    if fn not in __TA_FUNCTION_NAMES__:
#       if not fn.startswith("__") and fn == fn.upper():
#            print("not found ", fn)
if "_ta_lib" in globals():
    for func_name in _ta_lib.__TA_FUNCTION_NAMES__:  # type: ignore
        #    if func_name not in globals():
        #        print("new function", func_name)
        globals()[func_name] = getattr(_ta_lib, "stream_%s" % func_name)  # type: ignore
