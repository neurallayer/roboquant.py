from roboquant import Roboquant, EMACrossover, StandardTracker, YahooFeed

if __name__ == "__main__":
    feed = YahooFeed("JPM", "IBM", "F", start_date="2000-01-01")

    # split the feed timeframe in 5 equal parts
    timeframes = feed.timeframe().split(5)

    # run a back-test on each timeframe
    for timeframe in timeframes:
        rq = Roboquant(EMACrossover(13, 26))
        tracker = StandardTracker()
        rq.run(feed, tracker, timeframe)
        pnl = tracker.annualized_pnl() * 100
        mkt = tracker.get_market_return() * 100
        print(f"{timeframe}  portfolio-pnl = {pnl:5.2f}%    mkt-pnl = {mkt:5.2f}%")
