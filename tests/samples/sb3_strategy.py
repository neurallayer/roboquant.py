from stable_baselines3 import A2C
from stable_baselines3.common.policies import ActorCriticPolicy

from roboquant import run
from roboquant.feeds.yahoo import YahooFeed
from roboquant.ml.features import BarFeature, CombinedFeature, EquityFeature, PriceFeature, SMAFeature
from roboquant.ml.envs import Action2Signals, StrategyEnv
from roboquant.ml.strategies import SB3PolicyStrategy
from roboquant.traders import FlexTrader


def _train(symbols, path):
    yahoo = YahooFeed(*symbols, start_date="2000-01-01", end_date="2020-12-31")

    obs_feature = CombinedFeature(
            BarFeature(*symbols).returns(),
            SMAFeature(PriceFeature(*symbols), 5).returns(),
            SMAFeature(PriceFeature(*symbols), 10).returns(),
            SMAFeature(PriceFeature(*symbols), 20).returns(),
        ).normalize(20).cache()

    reward_feature = EquityFeature().returns().normalize(20)

    trader = FlexTrader(max_order_perc=0.2, min_order_perc=0.1)
    action_2_signals = Action2Signals(symbols)
    env = StrategyEnv(yahoo, obs_feature, reward_feature, action_2_signals, trader=trader)
    print(env)

    model = A2C("MlpPolicy", env, verbose=0)

    # Train the model
    model.learn(total_timesteps=100_000, progress_bar=True)

    policy = model.policy
    policy.save(path)
    return env


def _run(symbols, env, path):
    policy = ActorCriticPolicy.load(path)
    strategy = SB3PolicyStrategy.from_env(env, policy)
    feed = YahooFeed(*symbols, start_date="2021-01-01")
    account = run(feed, strategy, env.trader)
    print(account)


if __name__ == "__main__":
    SYMBOLS = ["IBM", "JPM", "MSFT", "BA", "AAPL", "AMZN"]
    PATH = "/tmp/trained_policy.zip"
    ENV = _train(SYMBOLS, PATH)
    _run(SYMBOLS, ENV, PATH)
