# %%
import logging

import roboquant as rq
from roboquant.alpaca import AlpacaHistoricStockFeed
from roboquant.feeds.feedutil import count_events
from roboquant.feeds.sqllitefeed import SQLFeed

from sb3_contrib import RecurrentPPO
from roboquant.ml.features import (
    CombinedFeature,
    QuoteFeature,
    EquityFeature,
    TimeDifference,
    DayOfWeekFeature,
    SMAFeature,
    PriceFeature,
)
from roboquant.ml.envs import Action2Signals, StrategyEnv
from roboquant.ml.strategies import SB3PolicyStrategy
from roboquant.traders.flextrader import FlexTrader

# %%
feed = SQLFeed("/tmp/apple_quotes.db", "quote")
symbol = "AAPL"
start_training = "2024-05-09T18:30:00Z"
end_training = "2024-05-09T18:35:00Z"
end_validation = "2024-05-14"

if not feed.exists():
    print("The retrieval of historical data will take some time....")
    alpaca_feed = AlpacaHistoricStockFeed()
    alpaca_feed.retrieve_quotes(symbol, start=start_training)
    print(alpaca_feed)

    # store it for later use
    feed.record(alpaca_feed)

# Run a backtest using the stored feed
print(feed)

obs_feature = (
    CombinedFeature(
        QuoteFeature(symbol).returns(),
        SMAFeature(PriceFeature(symbol), 20).returns(),
        SMAFeature(PriceFeature(symbol), 40).returns(),
        TimeDifference(),
        DayOfWeekFeature(),
    )
    .normalize(50)
    .cache()
)

reward_feature = EquityFeature().returns().normalize(20)

# %%
# logging.basicConfig()
# logging.getLogger("roboquant.ml.envs").setLevel(logging.INFO)
tf = rq.Timeframe.fromisoformat(start_training, end_training)
print(count_events(feed, tf))

trader = FlexTrader(max_order_perc=0.2, min_order_perc=0.01, max_position_perc=0.9, shorting=True)
action_2_signals = Action2Signals([symbol])
env = StrategyEnv(feed, obs_feature, reward_feature, action_2_signals, trader=trader, timeframe=tf)
model = RecurrentPPO("MlpLstmPolicy", env)

# %%
model.learn(total_timesteps=5_000_000, progress_bar=True)

# %%
logging.basicConfig()
logging.getLogger("roboquant").setLevel(logging.INFO)
tf = rq.Timeframe.fromisoformat(end_training, end_validation)
journal = rq.journals.BasicJournal()
obs_feature.reset()
strategy = SB3PolicyStrategy(obs_feature, action_2_signals, model.policy)

# %%
account = rq.run(feed, strategy, env.trader, journal=journal, timeframe=tf)

# %%
print(account)
print(journal)
