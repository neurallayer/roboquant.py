import os
import time

import roboquant as rq
from roboquant.traders.simpletrader import SimpleTrader



def _print(account: rq.Account, journal: rq.journals.BasicJournal, n_assets: int, load_time: float, runtime: float):
    print("", account, journal, sep="\n\n")

    candles = journal.items / 1_000_000.0
    throughput = candles / runtime

    # Print statistics
    print()
    print(f"load time  = {load_time:.1f}s")
    print(f"files      = {n_assets}")
    print(f"throughput = {n_assets / load_time:.0f} files/s")
    print(f"run time   = {runtime:.1f}s")
    print(f"candles    = {candles:.1f}M")
    print(f"throughput = {throughput:.2f}M candles/s")
    print()

def _run(feed: rq.Feed, journal: rq.journals.BasicJournal):
    strategy = rq.strategies.EMACrossover(13, 26)
    start = time.time()
    trader = SimpleTrader(50)
    account = rq.run(feed, strategy, trader = trader, journal=journal)
    return account, time.time() - start

def test_big_feed_daily():
    print("============ Daily Bars ============")
    start = time.time()
    path = os.path.expanduser("~/data/daily/us/nyse stocks/")
    feed = rq.feeds.CSVFeed.stooq_us_daily(path)
    load_time = time.time() - start

    journal = rq.journals.BasicJournal()
    account, runtime = _run(feed, journal)
    _print(account, journal, len(feed.assets()), load_time, runtime)

def test_big_feed_intraday():
    print("============ 5 min Bars ============")
    start = time.time()
    path = os.path.expanduser("~/data/5 min/us/nyse stocks/")
    feed = rq.feeds.CSVFeed.stooq_us_intraday(path)
    load_time = time.time() - start

    journal = rq.journals.BasicJournal()
    account, runtime = _run(feed, journal)
    _print(account, journal, len(feed.assets()), load_time, runtime)


if __name__ == "__main__":
    test_big_feed_daily()
    test_big_feed_intraday()
