---
kernelspec:
  name: python3
  display_name: Python 3
---


# Reinforcement Learning
The downside of training a typicall deep neural network is that you need to provide a label that represents the truth. So you need to know given a certain input what the ideal output should look like. 

With so much noise, it is very difficult to come up with good labels that allow the model to generalize well.

## Stable Baselines
Stable Baselines has good RL support with widely used, tested and reliable algorithms. This is very important since it is not easy to validate if a particular framework didn't make mistakes while implementing certain RL algorithms.

The version used by *roboquant* is Stable Baselines3 (SB3), with the implementations of reinforcement learning algorithms in PyTorch.

```{code} python
from sb3_contrib import RecurrentPPO
from sb3_contrib.common.recurrent.policies import RecurrentActorCriticPolicy
from roboquant import run
from roboquant.feeds.yahoofeed import YahooFeed
from roboquant.ai.features import BarFeature, EquityFeature, CombinedFeature, SMAFeature, PriceFeature
from roboquant.ai.rl import TradingEnv, SB3PolicyStrategy

# Create the feed
symbols = ["IBM", "JPM", "MSFT", "BA"]
feed = YahooFeed(*symbols, start_date="2000-01-01", end_date="2020-12-31")
assets = feed.assets()

# Create the features
obs_feature = CombinedFeature(
    BarFeature(*assets),
    SMAFeature(PriceFeature(*assets), period=20),
    SMAFeature(PriceFeature(*assets), period=10)
).returns().normalize(20)

reward_feature = EquityFeature().returns().normalize(20)

# Create the environment
env = TradingEnv(feed, obs_feature, reward_feature, assets)
model = RecurrentPPO("MlpLstmPolicy", env)

# Train the model and save the policy
model.learn(total_timesteps=20_000, progress_bar=False)
path = "/tmp/trained_recurrent_policy.zip"
model.policy.save(path)

# Load the trained policy as a strategy in roboquant
policy = RecurrentActorCriticPolicy.load(path)
strategy = SB3PolicyStrategy.from_env(env, policy)
feed = YahooFeed(*symbols, start_date="2021-01-01")
account = run(feed, strategy)
print(account)
```