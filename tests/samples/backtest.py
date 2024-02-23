from roboquant import Roboquant, EMACrossover, YahooFeed

if __name__ == "__main__":
    """Minimal back test scenario"""

    feed = YahooFeed("JPM", "IBM", "F", start_date="2000-01-01")
    rq = Roboquant(EMACrossover())
    account = rq.run(feed)
    print(account)
