# %% [markdown]
# This example shows how to use the ta-lib library to create a strategy that uses technical indicators
# from the TaLib library. The strategy combines the Relative Strength Index (RSI) and Bollinger Bands

# %%
from typing import override

import roboquant.ta as ta
import roboquant as rq
from roboquant.strategies import OHLCVBuffer, TaStrategy

# %%
class MyStrategy(TaStrategy):
    """Example using ta-lib to create a combined RSI/BollingerBand strategy:
    1. BUY if `RSI < 30 and close < lower band`
    2. SELL if `RSI > 70 and close > upper band`
    3. Otherwise do nothing
    """

    @override
    def _create_signal(self, asset: rq.Asset, ohlcv: OHLCVBuffer) -> rq.Signal | None:

        period = self.period - 1
        close_prices = ohlcv.close()
        rsi = ta.RSI(close_prices, timeperiod=period)

        upper, _, lower = ta.BBANDS(close_prices, timeperiod=period, nbdevup=2, nbdevdn=2)

        close: float = close_prices[-1]

        if rsi < 30 and close < lower:
            return rq.Signal.buy(asset)
        if rsi > 70 and close > upper:
            return rq.Signal.sell(asset)

        return None

# %%
feed = rq.feeds.YahooFeed("IBM", "AAPL", "JPM", "TSLA")

# ensure the size is enough for the used indicators
strategy = MyStrategy(14)

account = rq.run(feed, strategy)
print(account)
