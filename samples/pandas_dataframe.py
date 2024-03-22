# %%
import pandas as pd
import roboquant as rq

# %%
feed = rq.feeds.YahooFeed("IBM", start_date="2020-01-01")
ohlcv = rq.feeds.feedutil.get_ohlcv(feed, "IBM")

# %%
pd.DataFrame(ohlcv).set_index("Date")
