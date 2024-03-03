import matplotlib.pyplot as plt

import roboquant as rq

if __name__ == "__main__":

    # Plot historic price data
    feed = rq.feeds.YahooFeed("JPM", "IBM", "F", start_date="2010-01-01")
    _, axs = plt.subplots(2, figsize=(10, 10))
    feed.plot(axs[0], "JPM")
    feed.plot(axs[1], "IBM")
    plt.show()

    # run a single back-test
    strategy = rq.strategies.EMACrossover()
    journal = rq.journals.MetricsJournal.pnl()
    rq.run(feed, strategy, journal=journal)

    # Plot the equity curve with some customization
    journal.plot(plt, "pnl/equity", color="green", linestyle="--", linewidth=0.5)
    plt.show()

    # Plot all the recorded metrics
    metric_names = journal.get_metric_names()
    fig, axs = plt.subplots(3, 3, figsize=(15, 15))
    axs = axs.flatten()
    for ax, metric_name in zip(axs, metric_names):
        journal.plot(ax, metric_name)
    plt.show()
