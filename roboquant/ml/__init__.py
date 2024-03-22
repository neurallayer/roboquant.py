try:
    from gymnasium.envs.registration import register

    register(id="roboquant/StrategyEnv-v0", entry_point="roboquant.ml.envs:StrategyEnv")
    register(id="roboquant/TraderEnv-v0", entry_point="roboquant.ml.envs:TraderEnv")

except ImportError:
    pass
