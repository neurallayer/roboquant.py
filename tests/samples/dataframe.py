if __name__ == "__main__":
    # %%
    import pandas as pd
    import warnings
    import roboquant as rq

    warnings.simplefilter(action="ignore", category=FutureWarning)

    # %%
    feed = rq.feeds.YahooFeed("JPM", "IBM", "F", start_date="2010-01-01")
    ohlcv = rq.feeds.feedutil.get_symbol_ohlcv(feed, "IBM")

    # %%
    pd.DataFrame(ohlcv).set_index("Date")

# %%
