# %%
from alpaca.data.timeframe import TimeFrame
import roboquant as rq
from roboquant.alpaca import AlpacaHistoricStockFeed
from roboquant.feeds.sqllitefeed import SQLFeed

# %%
feed = SQLFeed("/tmp/test.db")

if not feed.exists():
    print("The retrieval of historic data will take some time....")
    alpaca_feed = AlpacaHistoricStockFeed()

    # Retrieve many years worth of 1-minute bars.
    alpaca_feed.retrieve_bars("AAPL", start = "2016-01-01", resolution=TimeFrame.Minute)  # type: ignore
    print(alpaca_feed)

    # store it for later use
    feed.record(alpaca_feed)

# Run a backtest using the stored feed
journal = rq.journals.BasicJournal()
print(feed.timeframe())
account = rq.run(feed, rq.strategies.EMACrossover(31, 101), journal = journal)
print(account)
print(journal)