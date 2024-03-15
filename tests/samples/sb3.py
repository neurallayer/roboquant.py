import logging
from stable_baselines3 import A2C

from roboquant.feeds.yahoo import YahooFeed
from roboquant.ml.features import PriceFeature, VolumeFeature, SMAFeature, PositionPNLFeature
from roboquant.ml.envs import TradingEnv


def run():
    # pylint: disable=unused-variable
    logging.basicConfig(level=logging.WARNING)

    symbols = ["IBM", "JPM"]
    yahoo = YahooFeed(*symbols, start_date="2000-01-01", end_date="2020-12-31")

    features = [
        PriceFeature(*symbols).returns(),
        VolumeFeature(*symbols).returns(),
        SMAFeature(PriceFeature(*symbols), 5).returns(),
        SMAFeature(PriceFeature(*symbols), 10).returns(),
        SMAFeature(PriceFeature(*symbols), 20).returns(),
        PositionPNLFeature(*symbols),
    ]

    env = TradingEnv(features, yahoo, symbols, warmup=50)
    env.calc_normalization(1000)
    print(env)

    model = A2C("MlpPolicy", env, verbose=1)

    # Train the model
    model.learn(log_interval=100, total_timesteps=10_000)

    # Run the trained model on out of sample data
    venv = model.get_env()
    assert venv is not None
    env.feed = YahooFeed(*symbols, start_date="2021-01-01")
    obs = venv.reset()
    done = False

    logging.getLogger("roboquant.ml.envs").setLevel(logging.DEBUG)
    while not done:
        action, _state = model.predict(obs, deterministic=True)  # type: ignore
        obs, reward, done, info = venv.step(action)

    print(env)


if __name__ == "__main__":
    run()
