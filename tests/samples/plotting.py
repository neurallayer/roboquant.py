import roboquant as rq
import matplotlib.pyplot as plt

if __name__ == "__main__":

    # Plot historic data
    feed = rq.feeds.YahooFeed("JPM", "IBM", "F", start_date="2000-01-01")
    feed.plot(plt)

    # Plot metrics
    strategy = rq.strategies.EMACrossover()
    journal = rq.journals.MetricsJournal.pnl()
    rq.run(feed, strategy, journal=journal)
    journal.plot(plt, "pnl/equity")

    # You can also plot live data
    feed = rq.feeds.TiingoLiveFeed(market="crypto")
    feed.subscribe("BTCUSDT")
    tf = rq.Timeframe.next(minutes=1)
    feed.plot(plt, timeframe=tf)
