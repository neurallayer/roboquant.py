# %%
import pandas as pd
import roboquant as rq
from dataclasses import asdict

pd.set_option('display.width', 1000)
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
positions = [asdict(asset) | asdict(position) for asset, position in account.positions.items()]
df = pd.json_normalize(positions)
print(df)

# %%
strategy = rq.strategies.EMACrossover()
account = rq.run(feed, strategy)
trades = [asdict(trade) for trade in account.trades]
df = pd.json_normalize(trades)
df.set_index("time")


# %%
strategy = rq.strategies.EMACrossover()
journal = rq.journals.MetricsJournal.pnl()
account = rq.run(feed, strategy, journal=journal)
equity = journal.get_metric("pnl/equity")
df = equity.to_dataframe(time_index=True)
print("Equity", df, sep="\n")
