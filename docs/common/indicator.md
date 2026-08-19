---
kernelspec:
  name: python3
  display_name: Python 3
---

# Indicators

(indicator_def)=
## Overview
Indicators are a thin wrapper around TA-Lib indicators that makes it easy
to use them within *roboquant*. TA-Lib comes with 200 indicators that are all
also available in *roboquant*. (Here)[https://ta-lib.org/functions/] you can find the complete list.

:::{note}
For those already familiar with the TA-Lib library: it is important to
note that *roboquant* uses the streaming version of the functions, even although
the function names don't start with `stream_`.
:::

## Use them in a Strategy
There are several places where you can use indicators, the most common
one is in your own strategy. 

```{code-cell} python
from roboquant.util.indicators import RSI, BBANDS
from roboquant import Asset, Signal
from roboquant.strategies import IndicatorStrategy
from roboquant.util import OHLCVBuffer

class MyStrategy(IndicatorStrategy):
    """
    Example using ta-lib to create a combined RSI/BollingerBand strategy:
    1. BUY if `RSI < 30 and close < lower band`
    2. SELL if `RSI > 70 and close > upper band`
    3. Otherwise do nothing
    """

    def _create_signal(self, asset: Asset, ohlcv: OHLCVBuffer) -> Signal | None:

        period = self.period - 1
        close_prices = ohlcv.close
        rsi = RSI(close_prices, timeperiod=period)

        upper, _, lower = BBANDS(close_prices, timeperiod=period)

        close: float = close_prices[-1]

        if rsi < 30 and close < lower:
            return rq.Signal.buy(asset)
        if rsi > 70 and close > upper:
            return rq.Signal.sell(asset)

        return None
```

## Indicatorfeature
But you can also use an indicator in a custom feature by subclassing
the `IndicatorFeature`.

```{code-cell} python
from roboquant.ai.features import IndicatorFeature

class RSIFeature(IndicatorFeature):
    """Example using TaLib to create a RSI feature"""

    def _calc(self, asset: Asset, ohlcv: OHLCVBuffer) -> float:
        return RSI(ohlcv.close, timeperiod=self.timeperiod - 1)

```
