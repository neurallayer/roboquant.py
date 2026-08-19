# %%
from roboquant.util.indicators import RSI
import roboquant as rq
from roboquant.common.asset import Asset
from roboquant.ai.features import IndicatorFeature
from roboquant.util.buffer import OHLCVBuffer

# %%
class RSIFeature(IndicatorFeature):
    """Example using TaLib to create a RSI feature"""

    def _calc(self, asset: Asset, ohlcv: OHLCVBuffer) -> float:
        return RSI(ohlcv.close, timeperiod=self.timeperiod - 1)

# %%
feed = rq.feeds.YahooFeed("IBM", "AAPL", start_date="2024-01-01", end_date="2024-02-01")
assets = feed.assets()
feature = RSIFeature(*assets, timeperiod=11)

for evt in feed.play():
    result = feature.calc(evt)
    print(assets[0].symbol, result[0], assets[1].symbol, result[1])
