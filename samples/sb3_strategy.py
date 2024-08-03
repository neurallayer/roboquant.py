# %%
from sb3_contrib import RecurrentPPO
from sb3_contrib.common.recurrent.policies import RecurrentActorCriticPolicy
from roboquant import run
from roboquant.feeds.yahoo import YahooFeed
from roboquant.ml.features import BarFeature, EquityFeature
from roboquant.ml.rl import TradingEnv, SB3PolicyStrategy

# %%
symbols = ["IBM", "JPM", "MSFT", "BA"]
path = "/tmp/trained_recurrent_policy.zip"

# %%
yahoo = YahooFeed(*symbols, start_date="2000-01-01", end_date="2020-12-31")
assets = yahoo.assets()

obs_feature = BarFeature(*assets).returns().normalize(20)
reward_feature = EquityFeature().returns().normalize(20)

env = TradingEnv(yahoo, obs_feature, reward_feature, assets)
model = RecurrentPPO("MlpLstmPolicy", env)

# %%
model.learn(total_timesteps=20_000, progress_bar=True)
model.policy.save(path)

# %%
policy = RecurrentActorCriticPolicy.load(path)
strategy = SB3PolicyStrategy.from_env(env, policy)
feed = YahooFeed(*symbols, start_date="2021-01-01")
account = run(feed, strategy)
print(account)
