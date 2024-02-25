import roboquant as rq


if __name__ == "__main__":
    feed = rq.feeds.YahooFeed("JPM", "IBM", "F", start_date="2000-01-01")

    # split the feed timeframe in 4 equal parts
    timeframes = feed.timeframe().split(4)

    # run a back-test on each timeframe
    for timeframe in timeframes:
        roboquant = rq.Roboquant(rq.strategies.EMACrossover(13, 26))
        journal = rq.journals.BasicJournal()
        roboquant.run(feed, journal, timeframe)
        pnl = journal.pnl * 100.0
        print(f"{timeframe}  pnl={pnl:5.2f}%")
