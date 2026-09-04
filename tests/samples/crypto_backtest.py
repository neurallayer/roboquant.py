# %% [markdown]
# This example shows how to use the crypto feed with a simple EMA Crossover strategy.

# %%
import ccxt
import roboquant as rq
from roboquant.feeds.crypto import CryptoFeed
from roboquant.common.monetary import USDT

# %%
exchange = ccxt.binance()
# exchange = ccxt.kraken()  # or any other exchange supported by ccxt
feed = CryptoFeed(exchange, "BTC/USDT", "ETH/USDT", start_date="2020-01-01 00:00:00", interval="1d")

# %%
for asset in feed.assets():
    feed.plot(asset)

# %%
strategy = rq.strategies.EMACrossover()
trader = rq.traders.FlexTrader(step_size="0.001", max_order_pct=0.2, max_position_pct=0.5, shorting=True)
broker = rq.brokers.SimBroker(deposit=10_000@USDT)
account = rq.run(feed, strategy, trader=trader, broker=broker)
print(account)

# %%
trades = sorted(account.trades, key=lambda t: t.pnl)
if trades:
    print(f"Biggest looser: {trades[0].pnl:.2f}")
    print(f"Biggest winner: {trades[-1].pnl:.2f}")
