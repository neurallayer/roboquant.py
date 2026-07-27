# %%
import roboquant as rq
from roboquant.feeds.parquetfeed import ParquetFeed

# %%

from cProfile import Profile
from pstats import Stats, SortKey

 # Profile the run to detect bottlenecks
with Profile() as profile:
    feed = ParquetFeed.us_stocks_10()
    for tf in feed.timeframe().sample(100, "365 days"):
        strategy = rq.strategies.EMACrossover()
        rq.run(feed, strategy, timeframe=tf)
    Stats(profile).sort_stats(SortKey.TIME).print_stats(.3)


