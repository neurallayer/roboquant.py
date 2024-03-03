import matplotlib.pyplot as plt

import roboquant as rq

if __name__ == "__main__":
    # Plot historic data
    feed = rq.feeds.YahooFeed("JPM", "IBM", "F", start_date="2000-01-01")
    feed.plot(plt)

    # Plot metrics
    strategy = rq.strategies.EMACrossover()
    journal = rq.journals.MetricsJournal.pnl()
    rq.run(feed, strategy, journal=journal)
    journal.plot(plt.subplot(), "pnl/equity", color="green")
    plt.show()
