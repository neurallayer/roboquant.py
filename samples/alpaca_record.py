# %%
import logging
from alpaca.data.timeframe import TimeFrame

import roboquant as rq
from roboquant.alpaca import AlpacaHistoricStockFeed
from roboquant.feeds.sqllitefeed import SQLFeed

from sb3_contrib import RecurrentPPO
from roboquant.ml.features import CombinedFeature, QuoteFeature, EquityFeature, TimeDifference
from roboquant.ml.envs import Action2Signals, StrategyEnv
from roboquant.ml.strategies import SB3PolicyStrategy
from roboquant.traders.flextrader import FlexTrader

# %%
feed = SQLFeed("/tmp/apple_quotes.db", "quote")
symbol = "AAPL"
start_training = "2024-05-08"
end_training = "2024-05-11"

if not feed.exists():
    print("The retrieval of historical data will take some time....")
    alpaca_feed = AlpacaHistoricStockFeed()
    alpaca_feed.retrieve_quotes(symbol, start = start_training)
    print(alpaca_feed)

    # store it for later use
    feed.record(alpaca_feed)

# Run a backtest using the stored feed
print(feed)

obs_feature = CombinedFeature(
    QuoteFeature(symbol).returns(),
    TimeDifference()
).normalize(20)

reward_feature = EquityFeature().returns().normalize(20)

#%%
# logging.basicConfig()
# logging.getLogger("roboquant").setLevel(logging.DEBUG)
tf = rq.Timeframe.fromisoformat(start_training, end_training)
trader = FlexTrader(max_order_perc=0.2, min_order_perc=0.1, shorting=True)
action_2_signals = Action2Signals([symbol])
env = StrategyEnv(feed, obs_feature, reward_feature, action_2_signals, trader=trader, timeframe=tf)
model = RecurrentPPO("MlpLstmPolicy", env)

# %%
model.learn(total_timesteps=20_000, progress_bar=True)

# %%
# logging.basicConfig()
# logging.getLogger("roboquant").setLevel(logging.DEBUG)
tf = rq.Timeframe.fromisoformat(end_training, "2025-01-01")
journal = rq.journals.BasicJournal()
strategy = SB3PolicyStrategy(obs_feature, action_2_signals, model.policy)

#%%
account = rq.run(feed, strategy, env.trader, journal=journal, timeframe=tf)

# %%
print(account)
print(journal)
