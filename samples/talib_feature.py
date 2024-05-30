# %%
import talib.stream as ta
import roboquant as rq
from roboquant.ml.features import TaFeature


# %%
class RSIFeature(TaFeature):
    """Example using talib to create an RSI feature"""

    def __init__(self, *symbols: str, timeperiod=10) -> None:
        self.timeperiod = timeperiod
        super().__init__(*symbols, history_size=timeperiod + 1)

    def _calc(self, symbol, ohlcv) -> float:
        close = ohlcv.close()
        return ta.RSI(close, timeperiod=self.timeperiod)  # type: ignore


# %%
feed = rq.feeds.YahooFeed("IBM", "AAPL", start_date="2024-01-01", end_date="2024-02-01")
feature = RSIFeature("IBM", "AAPL", timeperiod=15)
account = rq.Account()
channel = feed.play_background()

while evt := channel.get():
    result = feature.calc(evt, account)
    print(result)
