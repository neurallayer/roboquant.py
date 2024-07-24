# %%
import pandas as pd
import roboquant as rq

# %%
feed = rq.feeds.YahooFeed("IBM", start_date="2020-01-01")
ohlcv = feed.get_ohlcv(rq.Stock("IBM", "USD"))

# %%
pd.DataFrame(ohlcv).set_index("Date")
