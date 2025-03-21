# %%
# Make sure ta-lib 0.6.3 or higher is installed before running this sample
import roboquant.ta as ta
import roboquant as rq
from roboquant.strategies import OHLCVBuffer, TaStrategy

# %%
class MyStrategy(TaStrategy):
    """Example using ta-lib to create a combined RSI/BollingerBand strategy:
    - BUY if `RSI < 30 and close < lower band`
    - SELL if `RSI > 70 and close > upper band`
    - Otherwise do nothing
    """

    def process_asset(self, asset: rq.Asset, ohlcv: OHLCVBuffer):

        period = self.size - 1
        close_prices = ohlcv.close()
        rsi = ta.RSI(close_prices, timeperiod=period)

        upper, _, lower = ta.BBANDS(close_prices, timeperiod=period, nbdevup=2, nbdevdn=2)

        close = close_prices[-1]

        if rsi < 30 and close < lower:
            return rq.Signal.buy(asset)
        if rsi > 70 and close > upper:
            return rq.Signal.sell(asset)

        return None

# %%
feed = rq.feeds.YahooFeed("IBM", "AAPL")

# ensure the size is enough for the used indicators
strategy = MyStrategy(14)

account = rq.run(feed, strategy)
print(account)
