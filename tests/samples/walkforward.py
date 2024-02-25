import roboquant as rq


if __name__ == "__main__":
    feed = rq.feeds.YahooFeed("JPM", "IBM", "F", start_date="2000-01-01")

    # split the feed timeframe in 4 equal parts
    timeframes = feed.timeframe().split(4)

    # run a back-test on each timeframe
    for timeframe in timeframes:
        strategy = rq.strategies.EMACrossover(13, 26)
        account = rq.run(feed, strategy, timeframe=timeframe)
        print(f"{timeframe}  equity={account.equity:7_.2f}")
