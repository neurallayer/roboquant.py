# %%
from sb3_contrib import RecurrentPPO
from sb3_contrib.common.recurrent.policies import RecurrentActorCriticPolicy
from roboquant import run
from roboquant.alpaca.feed import AlpacaHistoricStockFeed
from roboquant.feeds.yahoo import YahooFeed
from roboquant.ml.features import EquityFeature, QuoteFeature
from roboquant.ml.envs import OrderMaker, TradingEnv, OrderWithLimitsMaker
from roboquant.ml.strategies import SB3PolicyStrategy

# %%
symbols = ["JPM"]

# %%
feed = AlpacaHistoricStockFeed()
feed.retrieve_quotes(*symbols, start="2024-05-01T18:00:00Z", end="2024-05-01T18:30:00Z")
print("events=",feed.events)

obs_feature = QuoteFeature(*symbols).returns().normalize(20)
reward_feature = EquityFeature().returns().normalize(20)

action_transformer = OrderMaker(symbols)
env = TradingEnv(feed, obs_feature, reward_feature, action_transformer)
model = RecurrentPPO("MlpLstmPolicy", env)

# %%
model.learn(total_timesteps=20_000, progress_bar=True)

# %%
strategy = SB3PolicyStrategy.from_env(env, model.policy)
feed = YahooFeed(*symbols, start_date="2021-01-01")
account = run(feed, strategy)
print(account)
