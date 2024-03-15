import numpy as np
from sb3_contrib import RecurrentPPO
from roboquant.feeds.yahoo import YahooFeed
from roboquant.ml.features import PriceFeature, VolumeFeature, PositionPNLFeature
from roboquant.ml.envs import TradingEnv


def run():
    # pylint: disable=unused-variable

    symbols = ["IBM", "JPM"]
    yahoo = YahooFeed(*symbols, start_date="2000-01-01", end_date="2020-12-31")

    features = [PriceFeature(*symbols).returns(), VolumeFeature(*symbols).returns(), PositionPNLFeature(*symbols)]

    env = TradingEnv(features, yahoo, symbols, warmup=50)
    env.calc_normalization(1000)
    print(env)

    model = RecurrentPPO("MlpLstmPolicy", env, verbose=1)

    # Train the model
    model.learn(log_interval=10, total_timesteps=10_000)

    # Run the trained model on out of sample data
    venv = model.get_env()
    assert venv is not None
    env.feed = YahooFeed(*symbols, start_date="2021-01-01")
    obs = venv.reset()
    done = np.zeros((1,), dtype=bool)
    state = None
    while not done:
        action, state = model.predict(obs, state=state, episode_start=done, deterministic=True)  # type: ignore
        print(action)
        obs, reward, done, info = venv.step(action)

    print(env)


if __name__ == "__main__":
    run()
