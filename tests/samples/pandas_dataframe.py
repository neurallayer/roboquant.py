# %%
import pandas as pd
import roboquant as rq

# %%
feed = rq.feeds.YahooFeed("IBM", start_date="2020-01-01")
df = feed.to_dataframe(rq.Stock("IBM"))
print("\nIBM Stock prices")
print(df)

# %%
feed = rq.feeds.YahooFeed("IBM", "JPM", "MSFT", "BTC-USD", "TSLA", "INTC", start_date="2020-01-01")
data = feed.to_dict(*feed.assets())
df = pd.DataFrame(data)
df.bfill(inplace=True)

print("\nAsset correlations")
print(df.corr())
