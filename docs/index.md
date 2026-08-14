
# Welcome

![roboquant logo](static/images/roboquant_header.png)

Roboquant is an open-source algorithmic trading platform written in Python. It is flexible, user-friendly and free to use. It is designed for anyone serious about algo-trading.

So whether you are a beginning retail trader or an established trading firm, roboquant can help you to develop robust and fully automated trading strategies.

## Usage
The following code snippet shows the steps to run a full back-test on a number of stocks.

```python
import roboquant as rq  # ⑴ 

feed = rq.feeds.YahooFeed("JPM", "IBM", "F", "TSLA")  # ⑵
strategy = rq.strategies.EMACrossover()  # ⑶ 
rq.run(feed, strategy)  # ⑷
```

1.  Import the roboquant package 
2.  Get historic market data from Yahoo Finance for 4 different stocks
3.  Create an instance of the strategy that you want to test
4.  Run the back test over all the market data in the feed using the strategy we just created

## Features
Below are some of the key features of *roboquant*:
- [x] fast back testing and live trading
- [x] AI/ML based strategies
- [x] strongly typed Python
- [x] market data feeds from CSV files, Yahoo Finance, Alpaca and many crypto exchanges
- [x] multi-currency trading
- [x] multi-asset trading with stocks, options, forex and crypto out of the box
- [x] larger-than-memory data feeds 
- [x] TaLib based indicators and strategies
- [x] plotting of prices and metrics
- [x] modular and extensible 

## Key Principles
- **Event-driven streaming** — Everything is built around {cl}`Event` objects produced lazily by feeds, supporting both backtesting and live trading with the same pipeline.
- **Pluggable Core** - All the core components can be replaced with different implementations
- **Good Develop Experience** — `run(feed, strategy)` works out of the box with sensible defaults, making the simplest back-test just a one-liner.
- **Immutable common types** — {cl}`Asset`, `Position`, `Order`, {cl}`Amount`, {cl}`Timeframe` and {cl}`Signal` are all immutable, reducing the risk of subtle bugs.
- **Multi-currency** — Built-in support via {cl}`Amount`, {cl}`Wallet`, and pluggable `CurrencyConverter` (ECB, static, one-to-one).

## License
Roboquant software itself is made available under the Apache 2.0 license. You can read more about the Apache 2.0 license on this page: https://www.apache.org/licenses/LICENSE-2.0.html

## Disclaimer
Absolutely no warranty is implied with this product. Use at your own risk. I provide no guarantee that it will be profitable, or that it won't lose all your money very quickly or does not contain bugs.

All financial trading offers the possibility of loss. Leveraged trading, may result in you losing all your money, and still owing more. Back tested results are no guarantee of future performance. I can take no responsibility for any losses caused by live trading using roboquant. Use at your own risk. I am not registered or authorized by any financial regulator.

