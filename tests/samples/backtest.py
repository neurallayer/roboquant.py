import roboquant as rq

if __name__ == "__main__":
    """Minimal back test scenario"""

    feed = rq.feeds.YahooFeed("JPM", "IBM", "F", start_date="2000-01-01")
    roboquant = rq.Roboquant(rq.strategies.EMACrossover())
    account = roboquant.run(feed)
    print(account)
