# %%
from roboquant.util.indicators import RSI
import roboquant as rq
from roboquant.common.asset import Asset
from roboquant.ai.features import IndicatorFeature
from roboquant.strategies.buffer import OHLCVBuffer

# %%
class RSIFeature(IndicatorFeature):
    """Example using TaLib to create a RSI feature"""

    def __init__(self, *assets: Asset, timeperiod:int) -> None:
        self.timeperiod = timeperiod
        super().__init__(*assets, period=timeperiod + 1)

    def _calc(self, asset: Asset, ohlcv: OHLCVBuffer) -> float:
        close = ohlcv.close()
        return RSI(close, timeperiod=self.timeperiod)

# %%
feed = rq.feeds.YahooFeed("IBM", "AAPL", start_date="2024-01-01", end_date="2024-02-01")
assets = feed.assets()
feature = RSIFeature(*assets, timeperiod=10)

for evt in feed.play():
    result = feature.calc(evt)
    print(assets[0].symbol, result[0], assets[1].symbol, result[1])
