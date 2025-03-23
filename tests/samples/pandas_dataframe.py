# %%
import pandas as pd
import roboquant as rq

# %%
feed = rq.feeds.YahooFeed("IBM", start_date="2020-01-01")
df = feed.to_dataframe(rq.Stock("IBM"))
print("IBM Stock prices", df, sep="\n")

# %%
feed = rq.feeds.YahooFeed("IBM", "JPM", "MSFT", "BTC-USD", "TSLA", "INTC", start_date="2020-01-01")
data = feed.to_dict(*feed.assets())
df = pd.DataFrame(data)
df.bfill(inplace=True)
print("Asset correlations", df.corr(), sep="\n")

# %%
strategy = rq.strategies.EMACrossover()
account = rq.run(feed, strategy)
df = pd.DataFrame(account.get_position_list())
print("Open positions", df, sep="\n")
