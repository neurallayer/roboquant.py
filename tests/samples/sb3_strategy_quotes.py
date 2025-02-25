# %%
import os
from sb3_contrib import RecurrentPPO
from roboquant import run
from roboquant.alpaca.feed import AlpacaHistoricStockFeed
from roboquant.asset import Stock
from roboquant.ml.features import EquityFeature, QuoteFeature
from roboquant.ml.rl import TradingEnv, SB3PolicyStrategy
from roboquant.timeframe import Timeframe

# %%
asset = Stock("JPM")
start = "2024-05-01T00:00:00Z"
border = "2024-05-01T20:00:00Z"
end = "2024-05-02T00:00:00Z"

assert start < border < end

# %%
api_key = os.environ["ALPACA_API_KEY"]
secret_key = os.environ["ALPACA_SECRET"]
feed = AlpacaHistoricStockFeed(api_key, secret_key)
feed.retrieve_quotes(asset.symbol, start=start, end=end)
print("feed timeframe=", feed.timeframe())

obs_feature = QuoteFeature(asset).returns().normalize(20)
reward_feature = EquityFeature().returns().normalize(20)

train_tf = Timeframe.fromisoformat(start, border)
env = TradingEnv(feed, obs_feature, reward_feature, [asset], timeframe=train_tf)
model = RecurrentPPO("MlpLstmPolicy", env)

# %%
steps = feed.count_events(timeframe=train_tf) * 5
model.learn(total_timesteps=steps, progress_bar=True)
model.policy.save("/tmp/jpm_quotes.zip")

# %%
strategy = SB3PolicyStrategy.from_env(env, model.policy)
test_tf = Timeframe.fromisoformat(border, end)
account = run(feed, strategy, timeframe=test_tf)
print(account)
