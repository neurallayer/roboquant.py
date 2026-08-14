---
kernelspec:
  name: python3
  display_name: Python 3
---

# Portfolio
(portfolio_def)=
A portfolio holds the open positions and is one of the attributes of the {cl}`Account`.

It behaves very much like a `dict[Asset, Position]`. So there is at most only one `Position` 
per asset. If a Position is zero, it will be removed from the Portfolio.

## Position
(position_def)=
A position is the net quantity of an {cl}`Asset` currently held, representing its market exposure and risk at any given moment.
Every time a trade is executed, the Position for that Asset is updated.

A Position contains 3 attributes:

- conntract size of Position
- average price paid
- latest known market price
  

