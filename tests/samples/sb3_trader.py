from sb3_contrib import RecurrentPPO
from sb3_contrib.common.recurrent.policies import RecurrentActorCriticPolicy
from roboquant import run
from roboquant.feeds.yahoo import YahooFeed
from roboquant.ml.features import BarFeature
from roboquant.ml.envs import TraderEnv
from roboquant.ml.strategies import SB3PolicyTrader


def _learn(path):
    symbols = ["IBM", "JPM", "MSFT", "BA"]
    yahoo = YahooFeed(*symbols, start_date="2000-01-01", end_date="2020-12-31")

    features = [
        BarFeature(*symbols).returns(),
    ]

    env = TraderEnv(features, yahoo, symbols)
    env.calc_normalization(1000)

    model = RecurrentPPO("MlpLstmPolicy", env, verbose=1)
    model.learn(log_interval=10, total_timesteps=100_000)
    model.policy.save(path)
    return env


def _run(env, path):
    policy = RecurrentActorCriticPolicy.load(path)
    trader = SB3PolicyTrader(env, policy)
    feed = YahooFeed(*env.symbols, start_date="2021-01-01")

    account = run(feed, trader=trader)
    print(account)


if __name__ == "__main__":
    PATH = "/tmp/trained_reccurrent_policy.zip"
    env1 = _learn(PATH)
    _run(env1, PATH)
