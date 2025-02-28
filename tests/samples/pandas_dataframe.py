# %%
import pandas as pd
import roboquant as rq

# %%
feed = rq.feeds.YahooFeed("IBM", start_date="2020-01-01")
ohlcv = feed.get_ohlcv(rq.Stock("IBM"))

# %%
columns = {idx: c for idx, c in enumerate("OHLCV")}
df = pd.DataFrame.from_dict(ohlcv, orient="index").rename(columns=columns)

print("IBM Stock prices:")
print(df)

# %%
feed = rq.feeds.YahooFeed("IBM", "JPM", "MSFT", "BTC-USD", "TSLA", "INTC", start_date="2020-01-01")
data = feed.to_dict(*feed.assets())
df = pd.DataFrame(data)
df.bfill(inplace=True)

print("\nAsset correlations:")
print(df.corr())
