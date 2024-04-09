# %%
import talib.stream as ta
import roboquant as rq

# %%
class MyStrategy(rq.strategies.TaStrategy):
    """Example using talib to create a combined RSI/BollingerBand strategy"""

    def _create_signal(self, symbol, ohlcv):
        close = ohlcv.close()

        rsi = ta.RSI(close, timeperiod=self.size-1)  # type: ignore

        upper, _, lower = ta.BBANDS(  # type: ignore
            close, timeperiod=self.size-1, nbdevup=2, nbdevdn=2
        )

        latest_price = close[-1]

        if rsi < 30 and latest_price < lower:
            return rq.Signal.buy(symbol)
        if rsi > 70 and latest_price > upper:
            return rq.Signal.sell(symbol)

        return None

# %%
feed = rq.feeds.YahooFeed("IBM", "AAPL")
strategy = MyStrategy(14)
account = rq.run(feed, strategy)
print(account)
# %%
