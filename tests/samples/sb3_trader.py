from sb3_contrib import RecurrentPPO
from sb3_contrib.common.recurrent.policies import RecurrentActorCriticPolicy
from roboquant import run
from roboquant.feeds.yahoo import YahooFeed
from roboquant.ml.features import BarFeature, EquityFeature
from roboquant.ml.envs import Action2Orders, TraderEnv
from roboquant.ml.strategies import SB3PolicyTrader


def _learn(symbols, path):
    yahoo = YahooFeed(*symbols, start_date="2000-01-01", end_date="2020-12-31")

    obs_feature = BarFeature(*symbols).returns().normalize(20)
    reward_feature = EquityFeature().returns().normalize(20)

    action_2_orders = Action2Orders(symbols)
    env = TraderEnv(yahoo, obs_feature, reward_feature, action_2_orders)
    model = RecurrentPPO("MlpLstmPolicy", env)
    model.learn(total_timesteps=20_000, progress_bar=True)
    model.policy.save(path)
    return env


def _run(symbols, env, path):
    policy = RecurrentActorCriticPolicy.load(path)
    trader = SB3PolicyTrader.from_env(env, policy)
    feed = YahooFeed(*symbols, start_date="2021-01-01")
    account = run(feed, trader=trader)
    print(account)


if __name__ == "__main__":
    SYMBOLS = ["IBM", "JPM", "MSFT", "BA"]
    PATH = "/tmp/trained_recurrent_policy.zip"
    ENV = _learn(SYMBOLS, PATH)
    _run(SYMBOLS, ENV, PATH)
