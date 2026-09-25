from cProfile import Profile
from pstats import Stats, SortKey

import roboquant as rq

feed = rq.feeds.RandomWalk(n_assets=500, n_events=365*20, start_date="2000-01-01T17:00:00Z")
print(f"timeframe: {feed.timeframe()}")
print(f"number of assets: {len(feed.assets())}")

def test_profile():
    print("\n\nRegular strategy\n##################################")
    strategy = rq.strategies.EMACrossover()
    journal = rq.journals.BasicJournal()

    # Profile the run to detect bottlenecks
    with Profile() as profile:
        rq.run(feed, strategy, journal=journal)
        print(f"\n{journal}")
        Stats(profile).sort_stats(SortKey.TIME).print_stats(.1, "roboquant")

if __name__ == "__main__":
    test_profile()
