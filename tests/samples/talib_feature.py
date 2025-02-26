# %%
# pylint: disable=no-member
import talib.stream as ta # type: ignore
import roboquant as rq
from roboquant.asset import Asset
from roboquant.ml.features import TaFeature
from roboquant.strategies.buffer import OHLCVBuffer

# %%
class RSIFeature(TaFeature):
    """Example using talib to create an RSI feature"""

    def __init__(self, *assets: Asset, timeperiod=10) -> None:
        self.timeperiod = timeperiod
        super().__init__(*assets, history_size=timeperiod + 1)

    def _calc(self, asset: Asset, ohlcv: OHLCVBuffer) -> float:
        close = ohlcv.close()
        return ta.RSI(close, timeperiod=self.timeperiod)  # type: ignore


# %%
feed = rq.feeds.YahooFeed("IBM", "AAPL", start_date="2024-01-01", end_date="2024-02-01")
feature = RSIFeature(*feed.assets(), timeperiod=15)
account = rq.Account()
channel = feed.play_background()

while evt := channel.get():
    result = feature.calc(evt)
    print(result)
