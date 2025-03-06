"""Set of stubs for the streaming version of the ta-lib indicators that makes them discoverable and typed."""

import numpy as np
from enum import Enum
from typing import Tuple
from numpy.typing import NDArray

try:
    import talib._ta_lib as _ta_lib  # type: ignore
except ImportError:
    pass

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


# Overlap Studies Functions


def BBANDS(
    real: NDArray[np.float64], timeperiod: int = 5, nbdevup: float = 2, nbdevdn: float = 2, matype: MA_Type = MA_Type.SMA
) -> Tuple[float, float, float]: ...


def DEMA(real: NDArray[np.float64], timeperiod: int = 30) -> float: ...


def EMA(real: NDArray[np.float64], timeperiod: int = 30) -> float: ...


def HT_TRENDLINE(real: NDArray[np.float64]) -> float: ...


def KAMA(real: NDArray[np.float64], timeperiod: int = 30) -> float: ...


def MA(real: NDArray[np.float64], timeperiod: int = 30, matype: MA_Type = MA_Type.SMA) -> float: ...


def MAMA(real: NDArray[np.float64], fastlimit: float = 0, slowlimit: float = 0) -> Tuple[float, float]: ...


def MAVP(
    real: NDArray[np.float64], periods: float, minperiod: int = 2, maxperiod: int = 30, matype: MA_Type = MA_Type.SMA
) -> float: ...


def MIDPOINT(real: NDArray[np.float64], timeperiod: int = 14) -> float: ...


def MIDPRICE(high: NDArray[np.float64], low: NDArray[np.float64], timeperiod: int = 14) -> float: ...


def SAR(high: NDArray[np.float64], low: NDArray[np.float64], acceleration: float = 0, maximum: float = 0) -> float: ...


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
) -> float: ...


def SMA(real: NDArray[np.float64], timeperiod: int = 30) -> float: ...


def T3(real: NDArray[np.float64], timeperiod: int = 5, vfactor: float = 0) -> float: ...


def TEMA(real: NDArray[np.float64], timeperiod: int = 30) -> float: ...


def TRIMA(real: NDArray[np.float64], timeperiod: int = 30) -> float: ...


def WMA(real: NDArray[np.float64], timeperiod: int = 30) -> float: ...


# Momentum Indicator Functions


def ADX(high: NDArray[np.float64], low: NDArray[np.float64], close: NDArray[np.float64], timeperiod: int = 14) -> float: ...


def ADXR(high: NDArray[np.float64], low: NDArray[np.float64], close: NDArray[np.float64], timeperiod: int = 14) -> float: ...


def APO(real: NDArray[np.float64], fastperiod: int = 12, slowperiod: int = 26, matype: MA_Type = MA_Type.SMA) -> float: ...


def AROON(high: NDArray[np.float64], low: NDArray[np.float64], timeperiod: int = 14) -> Tuple[float, float]: ...


def AROONOSC(high: NDArray[np.float64], low: NDArray[np.float64], timeperiod: int = 14) -> float: ...


def BOP(
    open: NDArray[np.float64], high: NDArray[np.float64], low: NDArray[np.float64], close: NDArray[np.float64]
) -> float: ...


def CCI(high: NDArray[np.float64], low: NDArray[np.float64], close: NDArray[np.float64], timeperiod: int = 14) -> float: ...


def CMO(real: NDArray[np.float64], timeperiod: int = 14) -> float: ...


def DX(high: NDArray[np.float64], low: NDArray[np.float64], close: NDArray[np.float64], timeperiod: int = 14) -> float: ...


def MACD(
    real: NDArray[np.float64], fastperiod: int = 12, slowperiod: int = 26, signalperiod: int = 9
) -> Tuple[float, float, float]: ...


def MACDEXT(
    real: NDArray[np.float64],
    fastperiod: int = 12,
    fastmatype: MA_Type = MA_Type.SMA,
    slowperiod: int = 26,
    slowmatype: MA_Type = MA_Type.SMA,
    signalperiod: int = 9,
    signalmatype: MA_Type = MA_Type.SMA,
) -> Tuple[float, float, float]: ...


def MACDFIX(real: NDArray[np.float64], signalperiod: int = 9) -> Tuple[float, float, float]: ...


def MFI(
    high: NDArray[np.float64],
    low: NDArray[np.float64],
    close: NDArray[np.float64],
    volume: NDArray[np.float64],
    timeperiod: int = 14,
) -> float: ...


def MINUS_DI(
    high: NDArray[np.float64], low: NDArray[np.float64], close: NDArray[np.float64], timeperiod: int = 14
) -> float: ...


def MINUS_DM(high: NDArray[np.float64], low: NDArray[np.float64], timeperiod: int = 14) -> float: ...


def MOM(real: NDArray[np.float64], timeperiod: int = 10) -> float: ...


def PLUS_DI(
    high: NDArray[np.float64], low: NDArray[np.float64], close: NDArray[np.float64], timeperiod: int = 14
) -> float: ...


def PLUS_DM(high: NDArray[np.float64], low: NDArray[np.float64], timeperiod: int = 14) -> float: ...


def PPO(real: NDArray[np.float64], fastperiod: int = 12, slowperiod: int = 26, matype: MA_Type = MA_Type.SMA) -> float: ...


def ROC(real: NDArray[np.float64], timeperiod: int = 10) -> float: ...


def ROCP(real: NDArray[np.float64], timeperiod: int = 10) -> float: ...


def ROCR(real: NDArray[np.float64], timeperiod: int = 10) -> float: ...


def ROCR100(real: NDArray[np.float64], timeperiod: int = 10) -> float: ...


def RSI(real: NDArray[np.float64], timeperiod: int = 14) -> float: ...


def STOCH(
    high: NDArray[np.float64],
    low: NDArray[np.float64],
    close: NDArray[np.float64],
    fastk_period: int = 5,
    slowk_period: int = 3,
    slowk_matype: MA_Type = MA_Type.SMA,
    slowd_period: int = 3,
    slowd_matype: MA_Type = MA_Type.SMA,
) -> Tuple[float, float]: ...


def STOCHF(
    high: NDArray[np.float64],
    low: NDArray[np.float64],
    close: NDArray[np.float64],
    fastk_period: int = 5,
    fastd_period: int = 3,
    fastd_matype: MA_Type = MA_Type.SMA,
) -> Tuple[float, float]: ...


def STOCHRSI(
    real: NDArray[np.float64],
    timeperiod: int = 14,
    fastk_period: int = 5,
    fastd_period: int = 3,
    fastd_matype: MA_Type = MA_Type.SMA,
) -> Tuple[float, float]: ...


def TRIX(real: NDArray[np.float64], timeperiod: int = 30) -> float: ...


def ULTOSC(
    high: NDArray[np.float64],
    low: NDArray[np.float64],
    close: NDArray[np.float64],
    timeperiod1: int = 7,
    timeperiod2: int = 14,
    timeperiod3: int = 28,
) -> float: ...


def WILLR(high: NDArray[np.float64], low: NDArray[np.float64], close: NDArray[np.float64], timeperiod: int = 14) -> float: ...


# Volume Indicator Functions


def AD(
    high: NDArray[np.float64], low: NDArray[np.float64], close: NDArray[np.float64], volume: NDArray[np.float64]
) -> float: ...


def ADOSC(
    high: NDArray[np.float64],
    low: NDArray[np.float64],
    close: NDArray[np.float64],
    volume: NDArray[np.float64],
    fastperiod: int = 3,
    slowperiod: int = 10,
) -> float: ...


def OBV(close: NDArray[np.float64], volume: NDArray[np.float64]) -> float: ...


# Volatility Indicator Functions


def ATR(high: NDArray[np.float64], low: NDArray[np.float64], close: NDArray[np.float64], timeperiod: int = 14) -> float: ...


def NATR(high: NDArray[np.float64], low: NDArray[np.float64], close: NDArray[np.float64], timeperiod: int = 14) -> float: ...


def TRANGE(high: NDArray[np.float64], low: NDArray[np.float64], close: NDArray[np.float64]) -> float: ...


# Price Transform Functions


def AVGPRICE(
    open: NDArray[np.float64], high: NDArray[np.float64], low: NDArray[np.float64], close: NDArray[np.float64]
) -> float: ...


def MEDPRICE(high: NDArray[np.float64], low: NDArray[np.float64]) -> float: ...


def TYPPRICE(high: NDArray[np.float64], low: NDArray[np.float64], close: NDArray[np.float64]) -> float: ...


def WCLPRICE(high: NDArray[np.float64], low: NDArray[np.float64], close: NDArray[np.float64]) -> float: ...


# Cycle Indicator Functions


def HT_DCPERIOD(real: NDArray[np.float64]) -> float: ...


def HT_DCPHASE(real: NDArray[np.float64]) -> float: ...


def HT_PHASOR(real: NDArray[np.float64]) -> Tuple[float, float]: ...


def HT_SINE(real: NDArray[np.float64]) -> Tuple[float, float]: ...


def HT_TRENDMODE(real: NDArray[np.float64]) -> Tuple[float, int]: ...


# Pattern Recognition Functions


def CDL2CROWS(
    open: NDArray[np.float64], high: NDArray[np.float64], low: NDArray[np.float64], close: NDArray[np.float64]
) -> int: ...


def CDL3BLACKCROWS(
    open: NDArray[np.float64], high: NDArray[np.float64], low: NDArray[np.float64], close: NDArray[np.float64]
) -> int: ...


def CDL3INSIDE(
    open: NDArray[np.float64], high: NDArray[np.float64], low: NDArray[np.float64], close: NDArray[np.float64]
) -> int: ...


def CDL3LINESTRIKE(
    open: NDArray[np.float64], high: NDArray[np.float64], low: NDArray[np.float64], close: NDArray[np.float64]
) -> int: ...


def CDL3OUTSIDE(
    open: NDArray[np.float64], high: NDArray[np.float64], low: NDArray[np.float64], close: NDArray[np.float64]
) -> int: ...


def CDL3STARSINSOUTH(
    open: NDArray[np.float64], high: NDArray[np.float64], low: NDArray[np.float64], close: NDArray[np.float64]
) -> int: ...


def CDL3WHITESOLDIERS(
    open: NDArray[np.float64], high: NDArray[np.float64], low: NDArray[np.float64], close: NDArray[np.float64]
) -> int: ...


def CDLABANDONEDBABY(
    open: NDArray[np.float64],
    high: NDArray[np.float64],
    low: NDArray[np.float64],
    close: NDArray[np.float64],
    penetration: float = 0,
) -> int: ...


def CDLADVANCEBLOCK(
    open: NDArray[np.float64], high: NDArray[np.float64], low: NDArray[np.float64], close: NDArray[np.float64]
) -> int: ...


def CDLBELTHOLD(
    open: NDArray[np.float64], high: NDArray[np.float64], low: NDArray[np.float64], close: NDArray[np.float64]
) -> int: ...


def CDLBREAKAWAY(
    open: NDArray[np.float64], high: NDArray[np.float64], low: NDArray[np.float64], close: NDArray[np.float64]
) -> int: ...


def CDLCLOSINGMARUBOZU(
    open: NDArray[np.float64], high: NDArray[np.float64], low: NDArray[np.float64], close: NDArray[np.float64]
) -> int: ...


def CDLCONCEALBABYSWALL(
    open: NDArray[np.float64], high: NDArray[np.float64], low: NDArray[np.float64], close: NDArray[np.float64]
) -> int: ...


def CDLCOUNTERATTACK(
    open: NDArray[np.float64], high: NDArray[np.float64], low: NDArray[np.float64], close: NDArray[np.float64]
) -> int: ...


def CDLDARKCLOUDCOVER(
    open: NDArray[np.float64],
    high: NDArray[np.float64],
    low: NDArray[np.float64],
    close: NDArray[np.float64],
    penetration: float = 0,
) -> int: ...


def CDLDOJI(
    open: NDArray[np.float64], high: NDArray[np.float64], low: NDArray[np.float64], close: NDArray[np.float64]
) -> int: ...


def CDLDOJISTAR(
    open: NDArray[np.float64], high: NDArray[np.float64], low: NDArray[np.float64], close: NDArray[np.float64]
) -> int: ...


def CDLDRAGONFLYDOJI(
    open: NDArray[np.float64], high: NDArray[np.float64], low: NDArray[np.float64], close: NDArray[np.float64]
) -> int: ...


def CDLENGULFING(
    open: NDArray[np.float64], high: NDArray[np.float64], low: NDArray[np.float64], close: NDArray[np.float64]
) -> int: ...


def CDLEVENINGDOJISTAR(
    open: NDArray[np.float64],
    high: NDArray[np.float64],
    low: NDArray[np.float64],
    close: NDArray[np.float64],
    penetration: float = 0,
) -> int: ...


def CDLEVENINGSTAR(
    open: NDArray[np.float64],
    high: NDArray[np.float64],
    low: NDArray[np.float64],
    close: NDArray[np.float64],
    penetration: float = 0,
) -> int: ...


def CDLGAPSIDESIDEWHITE(
    open: NDArray[np.float64], high: NDArray[np.float64], low: NDArray[np.float64], close: NDArray[np.float64]
) -> int: ...


def CDLGRAVESTONEDOJI(
    open: NDArray[np.float64], high: NDArray[np.float64], low: NDArray[np.float64], close: NDArray[np.float64]
) -> int: ...


def CDLHAMMER(
    open: NDArray[np.float64], high: NDArray[np.float64], low: NDArray[np.float64], close: NDArray[np.float64]
) -> int: ...


def CDLHANGINGMAN(
    open: NDArray[np.float64], high: NDArray[np.float64], low: NDArray[np.float64], close: NDArray[np.float64]
) -> int: ...


def CDLHARAMI(
    open: NDArray[np.float64], high: NDArray[np.float64], low: NDArray[np.float64], close: NDArray[np.float64]
) -> int: ...


def CDLHARAMICROSS(
    open: NDArray[np.float64], high: NDArray[np.float64], low: NDArray[np.float64], close: NDArray[np.float64]
) -> int: ...


def CDLHIGHWAVE(
    open: NDArray[np.float64], high: NDArray[np.float64], low: NDArray[np.float64], close: NDArray[np.float64]
) -> int: ...


def CDLHIKKAKE(
    open: NDArray[np.float64], high: NDArray[np.float64], low: NDArray[np.float64], close: NDArray[np.float64]
) -> int: ...


def CDLHIKKAKEMOD(
    open: NDArray[np.float64], high: NDArray[np.float64], low: NDArray[np.float64], close: NDArray[np.float64]
) -> int: ...


def CDLHOMINGPIGEON(
    open: NDArray[np.float64], high: NDArray[np.float64], low: NDArray[np.float64], close: NDArray[np.float64]
) -> int: ...


def CDLIDENTICAL3CROWS(
    open: NDArray[np.float64], high: NDArray[np.float64], low: NDArray[np.float64], close: NDArray[np.float64]
) -> int: ...


def CDLINNECK(
    open: NDArray[np.float64], high: NDArray[np.float64], low: NDArray[np.float64], close: NDArray[np.float64]
) -> int: ...


def CDLINVERTEDHAMMER(
    open: NDArray[np.float64], high: NDArray[np.float64], low: NDArray[np.float64], close: NDArray[np.float64]
) -> int: ...


def CDLKICKING(
    open: NDArray[np.float64], high: NDArray[np.float64], low: NDArray[np.float64], close: NDArray[np.float64]
) -> int: ...


def CDLKICKINGBYLENGTH(
    open: NDArray[np.float64], high: NDArray[np.float64], low: NDArray[np.float64], close: NDArray[np.float64]
) -> int: ...


def CDLLADDERBOTTOM(
    open: NDArray[np.float64], high: NDArray[np.float64], low: NDArray[np.float64], close: NDArray[np.float64]
) -> int: ...


def CDLLONGLEGGEDDOJI(
    open: NDArray[np.float64], high: NDArray[np.float64], low: NDArray[np.float64], close: NDArray[np.float64]
) -> int: ...


def CDLLONGLINE(
    open: NDArray[np.float64], high: NDArray[np.float64], low: NDArray[np.float64], close: NDArray[np.float64]
) -> int: ...


def CDLMARUBOZU(
    open: NDArray[np.float64], high: NDArray[np.float64], low: NDArray[np.float64], close: NDArray[np.float64]
) -> int: ...


def CDLMATCHINGLOW(
    open: NDArray[np.float64], high: NDArray[np.float64], low: NDArray[np.float64], close: NDArray[np.float64]
) -> int: ...


def CDLMATHOLD(
    open: NDArray[np.float64],
    high: NDArray[np.float64],
    low: NDArray[np.float64],
    close: NDArray[np.float64],
    penetration: float = 0,
) -> int: ...


def CDLMORNINGDOJISTAR(
    open: NDArray[np.float64],
    high: NDArray[np.float64],
    low: NDArray[np.float64],
    close: NDArray[np.float64],
    penetration: float = 0,
) -> int: ...


def CDLMORNINGSTAR(
    open: NDArray[np.float64],
    high: NDArray[np.float64],
    low: NDArray[np.float64],
    close: NDArray[np.float64],
    penetration: float = 0,
) -> int: ...


def CDLONNECK(
    open: NDArray[np.float64], high: NDArray[np.float64], low: NDArray[np.float64], close: NDArray[np.float64]
) -> int: ...


def CDLPIERCING(
    open: NDArray[np.float64], high: NDArray[np.float64], low: NDArray[np.float64], close: NDArray[np.float64]
) -> int: ...


def CDLRICKSHAWMAN(
    open: NDArray[np.float64], high: NDArray[np.float64], low: NDArray[np.float64], close: NDArray[np.float64]
) -> int: ...


def CDLRISEFALL3METHODS(
    open: NDArray[np.float64], high: NDArray[np.float64], low: NDArray[np.float64], close: NDArray[np.float64]
) -> int: ...


def CDLSEPARATINGLINES(
    open: NDArray[np.float64], high: NDArray[np.float64], low: NDArray[np.float64], close: NDArray[np.float64]
) -> int: ...


def CDLSHOOTINGSTAR(
    open: NDArray[np.float64], high: NDArray[np.float64], low: NDArray[np.float64], close: NDArray[np.float64]
) -> int: ...


def CDLSHORTLINE(
    open: NDArray[np.float64], high: NDArray[np.float64], low: NDArray[np.float64], close: NDArray[np.float64]
) -> int: ...


def CDLSPINNINGTOP(
    open: NDArray[np.float64], high: NDArray[np.float64], low: NDArray[np.float64], close: NDArray[np.float64]
) -> int: ...


def CDLSTALLEDPATTERN(
    open: NDArray[np.float64], high: NDArray[np.float64], low: NDArray[np.float64], close: NDArray[np.float64]
) -> int: ...


def CDLSTICKSANDWICH(
    open: NDArray[np.float64], high: NDArray[np.float64], low: NDArray[np.float64], close: NDArray[np.float64]
) -> int: ...


def CDLTAKURI(
    open: NDArray[np.float64], high: NDArray[np.float64], low: NDArray[np.float64], close: NDArray[np.float64]
) -> int: ...


def CDLTASUKIGAP(
    open: NDArray[np.float64], high: NDArray[np.float64], low: NDArray[np.float64], close: NDArray[np.float64]
) -> int: ...


def CDLTHRUSTING(
    open: NDArray[np.float64], high: NDArray[np.float64], low: NDArray[np.float64], close: NDArray[np.float64]
) -> int: ...


def CDLTRISTAR(
    open: NDArray[np.float64], high: NDArray[np.float64], low: NDArray[np.float64], close: NDArray[np.float64]
) -> int: ...


def CDLUNIQUE3RIVER(
    open: NDArray[np.float64], high: NDArray[np.float64], low: NDArray[np.float64], close: NDArray[np.float64]
) -> int: ...


def CDLUPSIDEGAP2CROWS(
    open: NDArray[np.float64], high: NDArray[np.float64], low: NDArray[np.float64], close: NDArray[np.float64]
) -> int: ...


def CDLXSIDEGAP3METHODS(
    open: NDArray[np.float64], high: NDArray[np.float64], low: NDArray[np.float64], close: NDArray[np.float64]
) -> int: ...


# Statistic Functions


def BETA(real0: NDArray[np.float64], real1: NDArray[np.float64], timeperiod: int = 5) -> float: ...


def CORREL(real0: NDArray[np.float64], real1: NDArray[np.float64], timeperiod: int = 30) -> float: ...


def LINEARREG(real: NDArray[np.float64], timeperiod: int = 14) -> float: ...


def LINEARREG_ANGLE(real: NDArray[np.float64], timeperiod: int = 14) -> float: ...


def LINEARREG_INTERCEPT(real: NDArray[np.float64], timeperiod: int = 14) -> float: ...


def LINEARREG_SLOPE(real: NDArray[np.float64], timeperiod: int = 14) -> float: ...


def STDDEV(real: NDArray[np.float64], timeperiod: int = 5, nbdev: float = 1) -> float: ...


def TSF(real: NDArray[np.float64], timeperiod: int = 14) -> float: ...


def VAR(real: NDArray[np.float64], timeperiod: int = 5, nbdev: float = 1) -> float: ...


# Math Transform Functions


def ACOS(real: NDArray[np.float64]) -> float: ...


def ASIN(real: NDArray[np.float64]) -> float: ...


def ATAN(real: NDArray[np.float64]) -> float: ...


def CEIL(real: NDArray[np.float64]) -> float: ...


def COS(real: NDArray[np.float64]) -> float: ...


def COSH(real: NDArray[np.float64]) -> float: ...


def EXP(real: NDArray[np.float64]) -> float: ...


def FLOOR(real: NDArray[np.float64]) -> float: ...


def LN(real: NDArray[np.float64]) -> float: ...


def LOG10(real: NDArray[np.float64]) -> float: ...


def SIN(real: NDArray[np.float64]) -> float: ...


def SINH(real: NDArray[np.float64]) -> float: ...


def SQRT(real: NDArray[np.float64]) -> float: ...


def TAN(real: NDArray[np.float64]) -> float: ...


def TANH(real: NDArray[np.float64]) -> float: ...


# Math Operator Functions


def ADD(real0: NDArray[np.float64], real1: NDArray[np.float64]) -> float: ...


def DIV(real0: NDArray[np.float64], real1: NDArray[np.float64]) -> float: ...


def MAX(real: NDArray[np.float64], timeperiod: int = 30) -> float: ...


def MAXINDEX(real: NDArray[np.float64], timeperiod: int = 30) -> int: ...


def MIN(real: NDArray[np.float64], timeperiod: int = 30) -> float: ...


def MININDEX(real: NDArray[np.float64], timeperiod: int = 30) -> int: ...


def MINMAX(real: NDArray[np.float64], timeperiod: int = 30) -> Tuple[float, float]: ...


def MINMAXINDEX(real: NDArray[np.float64], timeperiod: int = 30) -> Tuple[float, float]: ...


def MULT(real0: NDArray[np.float64], real1: NDArray[np.float64]) -> float: ...


def SUB(real0: NDArray[np.float64], real1: NDArray[np.float64]) -> float: ...


def SUM(real: NDArray[np.float64], timeperiod: int = 30) -> float: ...


# ################
# missed ones
# ###############


def ACCBANDS(
    open: NDArray[np.float64], high: NDArray[np.float64], low: NDArray[np.float64], timeperiod: int = 20
) -> Tuple[float, float, float]: ...


def AVGDEV(real: NDArray[np.float64], timeperiod: int = 14) -> float: ...


def IMI(open: NDArray[np.float64], close: NDArray[np.float64], timeperiod: int = 14) -> float: ...


# for fn in dict(globals()):
#    if fn not in __TA_FUNCTION_NAMES__:
#       if not fn.startswith("__") and fn == fn.upper():
#            print("not found ", fn)
if "_ta_lib" in globals():
    for func_name in _ta_lib.__TA_FUNCTION_NAMES__:  # type: ignore
        #    if func_name not in globals():
        #        print("new function", func_name)
        globals()[func_name] = getattr(_ta_lib, "stream_%s" % func_name)  # type: ignore
