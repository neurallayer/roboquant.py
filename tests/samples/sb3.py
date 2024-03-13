from gymnasium.wrappers.frame_stack import FrameStack
from stable_baselines3 import A2C

from roboquant.feeds.yahoo import YahooFeed
from roboquant.ml.features import PriceFeature, VolumeFeature, SMAFeature, PositionPNLFeature
from roboquant.ml.gymenv import TradingEnv


def run():
    # pylint: disable=unused-variable
    yahoo = YahooFeed("IBM", "JPM", start_date="2000-01-01", end_date="2020-12-31")

    features = [
        PriceFeature("IBM", "JPM").returns(),
        VolumeFeature("IBM", "JPM").returns(),
        SMAFeature(PriceFeature("JPM"), 10).returns(),
        PositionPNLFeature("IBM", "JPM"),
    ]

    trading = TradingEnv(features, yahoo, yahoo.symbols, warmup=20)
    trading.calc_normalization(1000)

    env = FrameStack(trading, 10)
    model = A2C("MlpPolicy", env, verbose=1)

    # Train the model
    model.learn(total_timesteps=1_000_000)

    # Run the trained model on out of sample data
    venv = model.get_env()
    assert venv is not None
    trading.feed = YahooFeed("IBM", "JPM", start_date="2021-01-01")
    obs = venv.reset()
    done = False
    while not done:
        action, _state = model.predict(obs, deterministic=True)  # type: ignore
        obs, reward, done, info = venv.step(action)

    print(trading.last_equity)


if __name__ == "__main__":
    run()
