# %%
# pylint: disable=no-member
import talib.stream as ta
import roboquant as rq
from roboquant.signal import Signal
from roboquant.strategies import OHLCVBuffer, TaStrategy


# %%
class MyStrategy(TaStrategy):
    """Example using talib to create a combined RSI/BollingerBand strategy"""

    def process_asset(self, asset: rq.Asset, ohlcv: OHLCVBuffer):

        close = ohlcv.close()

        rsi = ta.RSI(close, timeperiod=self.size - 1)  # type: ignore

        upper, _, lower = ta.BBANDS(close, timeperiod=self.size - 1, nbdevup=2, nbdevdn=2)  # type: ignore

        latest_price = close[-1]

        if rsi < 30 and latest_price < lower:
            return Signal.buy(asset)
        if rsi > 70 and latest_price > upper:
            return Signal.sell(asset)

        return None


# %%
feed = rq.feeds.YahooFeed("IBM", "AAPL")
strategy = MyStrategy(14)
account = rq.run(feed, strategy)
print(account)
