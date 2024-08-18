# %%
import pandas as pd
import roboquant as rq

# %%
feed = rq.feeds.YahooFeed("IBM", start_date="2020-01-01")
ohlcv = feed.get_ohlcv(rq.Stock("IBM"))

# %%
pd.DataFrame(ohlcv).set_index("Date")

# %%
feed = rq.feeds.YahooFeed("IBM", "JPM", "MSFT", "BTC-USD", "TSLA", "INTC", start_date="2015-01-01")
data = feed.to_dict(*feed.assets())
df = pd.DataFrame(data).bfill()
df.corr()
