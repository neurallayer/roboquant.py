from gymnasium.envs.registration import register

register(id="roboquant/TradingStrategy-v0", entry_point="roboquant.ml.envs:StrategyEnv")
