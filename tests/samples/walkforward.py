from roboquant import Roboquant, EMACrossover, EquityTracker, YahooFeed


if __name__ == "__main__":
    feed = YahooFeed("JPM", "IBM", "F", start_date="2000-01-01")

    # split the feed timeframe in 4 equal parts
    timeframes = feed.timeframe().split(4)

    # run a back-test on each timeframe
    for timeframe in timeframes:
        rq = Roboquant(EMACrossover(13, 26))
        tracker = EquityTracker()
        rq.run(feed, tracker, timeframe)
        pnl = tracker.pnl(annualized=True) * 100
        print(f"{timeframe}  portfolio-pnl = {pnl:5.2f}%")
