"""roboquant.ai package.
Provides a number of AI related classes and methods to support the development of AI based trading strategies.
It relies on Stable Baselines3 for reinforcement learning and PyTorch for deep learning.

This package also introduces the concept of Features, which are the building blocks for roboquant machine learning models
and can be used to extract relevant information. The ones included by default are either based on an `Event` or an `Account`.
Typically Event features are used for input data and Account features are used for reward/label/output data.
"""
