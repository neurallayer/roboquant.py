---
kernelspec:
  name: python3
  display_name: Python 3
---

# Introduction
One of the compelling arguments to use Python for algo-trading, is the relative ease of using AI and machine learning as part of your solution.
That being said, if you are new to Python development, it might still seem a bit daunting at first.

The required dependencies can be installed: 

```bash
pip install roboquant[ai]
```

Please note that these dependencies are large, so it can take some time before
they are downloaded and installed.


## Feature
A feature can work on an event or the account and serves as both input and labels for ML models. 
Features are meant to being used in an event based system, so they don't introduce the risk 
of future bias as so often seen in other solutions.

There are many commonly used features already provided out of the box.

## PyTorch
Roboquant uses the excellent PyTorch framework as a basis for the included neural-networks based strategies. 
It comes with many popular network archtecture layers out of the box and has a Pythonic API, making it
easier to get started. 


## Reinforcement Learning
The downside of training a typicall deep neural network is that you need a label that represents the truth.
So you need to know given a certain input what the ideal output should look like. With so much noise, it is
difficult to come up with good labels.

RL sidesteps the need for explicit labels entirely. Instead of learning from a static dataset of "correct" answers,
an RL agent learns by interacting with the market environment and receiving a **reward signal** — typically
based on P&L, Sharpe ratio, or risk-adjusted returns. This makes RL a natural fit for algo-trading because:

- **No ground truth required.** You don't need to know the "right" trade beforehand; the agent discovers
  profitable strategies through trial and error.
- **Handles sequential decision-making.** Trading is inherently sequential — today's action affects
  tomorrow's position, capital, and risk exposure. RL explicitly models this temporal dependency.
- **Optimises for long-term objectives.** Unlike supervised learning which minimises prediction error
  on individual samples, RL can be tuned to maximise cumulative returns, a more direct proxy for
  trading success.
- **Adapts to non-stationary environments.** Markets evolve; RL agents can be designed to continuously
  learn and adapt as new data arrives, rather than relying on a frozen model trained once.

The downside is that the setup and configuration to have a RL based solution in place is more challenging.
