from stable_baselines3 import A2C
from stable_baselines3.common.policies import ActorCriticPolicy

from roboquant import run
from roboquant.feeds.yahoo import YahooFeed
from roboquant.ml.features import BarFeature, CombinedFeature, EquityFeature, PriceFeature, SMAFeature
from roboquant.ml.envs import StrategyEnv
from roboquant.ml.strategies import SB3PolicyStrategy
from roboquant.traders import FlexTrader


def _train(path):
    symbols = ["IBM", "JPM", "MSFT", "BA", "AAPL", "AMZN"]
    yahoo = YahooFeed(*symbols, start_date="2000-01-01", end_date="2020-12-31")

    obs_feature = CombinedFeature(
            BarFeature(*symbols).returns(),
            SMAFeature(PriceFeature(*symbols), 5).returns(),
            SMAFeature(PriceFeature(*symbols), 10).returns(),
            SMAFeature(PriceFeature(*symbols), 20).returns(),
        ).normalize(20).cache()

    reward_feature = EquityFeature().normalize(20)

    trader = FlexTrader(max_order_perc=0.2, min_order_perc=0.1)
    env = StrategyEnv(obs_feature, reward_feature, feed=yahoo, rating_symbols=symbols, trader=trader)
    print(env)

    model = A2C("MlpPolicy", env, verbose=1)

    # Train the model
    model.learn(log_interval=10_000, total_timesteps=100_000)

    policy = model.policy
    policy.save(path)
    return env


def _run(env, path):
    policy = ActorCriticPolicy.load(path)
    strategy = SB3PolicyStrategy(env, policy)
    feed = YahooFeed(*env.symbols, start_date="2021-01-01")
    account = run(feed, strategy, env.trader)
    print(account)


if __name__ == "__main__":
    PATH = "/tmp/trained_policy.zip"
    env1 = _train(PATH)
    _run(env1, PATH)
