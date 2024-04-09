# %%
import talib.stream as ta
import roboquant as rq
from roboquant.ml.features import TaFeature

# %%
class MyFeature(TaFeature):
    """Example using talib to create a combined RSI/BollingerBand strategy"""


    def __init__(self, *symbols: str) -> None:
        super().__init__(*symbols, history_size=10)

    def _calc(self, symbol, ohlcv) -> float:
        close = ohlcv.close()
        return ta.RSI(close, timeperiod=9)  # type: ignore

# %%
feature = MyFeature()