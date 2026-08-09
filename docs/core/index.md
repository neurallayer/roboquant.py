---
kernelspec:
  name: python3
  display_name: Python 3
---

# Core Components
The library is built around a clean separation of five orthogonal concerns:

| Component | Responsibility |
|-----------|---------------|
| **[Feed](feed.md)** | Provides (market) data events |
| **[Strategy](strategy.md)** | Generates trading signals from events |
| **[Trader](trader.md)** | Converts signals into orders (risk/sizing) |
| **[Broker](broker.md)** | Executes orders, maintains account state |
| **[Journal](journal.md)** | Logs/tracks every step (read-only) |

Each component is an abstract base class with pluggable implementations, making every part of the pipeline independently swappable.

:::{note} Strategy/Trader Separation
This is an important design choice:

- **Strategies** produce signals from market data only. They implement `create_signals(event) -> list[Signal]` and have no access to account state. This keeps them pure, testable, and reusable across any trader configuration.
- **Traders** implement `create_orders(signals, event, account) -> list[Order]` and are responsible for risk management, position sizing, and order construction.
:::


## The Run Loop
The core of the system is the [roboquant.run()](run.md) function, which connects all components in a streaming event loop. At a high level, the run loop looks like this:

```mermaid
flowchart LR
 
    Feed["Feed"]
    Strategy["Strategy"]
    Trader["Trader"]
    Broker["Broker"]
    Journal["Journal"]
    
    Feed -- event --> Strategy -- signals --> Trader -- orders --> Broker -- account --> Journal 
```


