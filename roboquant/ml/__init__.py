from gymnasium.envs.registration import register

register(id="roboquant/Trading-v0", entry_point="roboquant.ml.envs:TradingEnv")
