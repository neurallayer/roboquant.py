
# %% [markdown]
# Report enables to publication of mathplotlib charts. They can be saved
# as a single self-contained PDF file or HTML file.
# Besides charts, you can also add Pandas DataFrames to the publication
#
# This might be better solution if you are running a long back test and
# don't want to wait for the result.
#
#

# %%
import roboquant as rq

# %%
feed = rq.feeds.YahooFeed.us_stocks_10()
strategy = rq.strategies.EMACrossover(26, 50)
journal = rq.journals.MetricsJournal.pnl()
account = rq.run(feed, strategy, journal=journal)

report = rq.journals.Report()

for asset in feed.assets():
    feed.plot(asset, trades=account.trades, linewidth=0.5, color="grey")
    report.add_current_figure()

journal.plot("pnl/equity")
report.add_current_figure()

df = account.trades_to_dataframe().round(2)
top_trades = df.sort_values("pnl", ascending=False)[:25]
report.add_df(top_trades, "top 25 winners")

down_trades = df.sort_values("pnl", ascending=True)[:25]
report.add_df(down_trades, "top 25 losers")

report.save_as_pdf("/tmp/report.pdf")
report.save_as_html("/tmp/report.html")
