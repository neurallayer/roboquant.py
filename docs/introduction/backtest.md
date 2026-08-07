---
kernelspec:
  name: python3
  display_name: Python 3
---

# Getting started
This page shows how to run a simple back test using *roboquant*. A back test is a simulation of a trading strategy using historical data. It allows you to see how the strategy would have performed in the past, and can help you to identify any potential issues with the strategy before you start trading with real money.

There are many more advanced features in *roboquant*, like live trading, paper trading, multi-asset trading, multi-currency trading, and more. But this page is meant to be a simple introduction to the basic concepts of back testing.

## Import
We always start with the import of the roboquant package. This package contains all the classes and functions that we need to run a back test.

```{code-cell} python
import roboquant as rq
```

```{code-cell} python
:tags: [remove-input]
rq.set_dark_style()
```

You can check the version and other information about the installed *roboquant* package by running the following command:

```{code-cell} python
rq.info()
```

## Feed
For a back test we'll need historic data. *Roboquant* uses the concept of a `Feed` to provide this data.
There are several `Feed` providers included, like the Yahoo Finance one used in this example.

It is free to use without an API key and provides data for a large number of stocks, ETFs, indices and more. The data is provided in the form of `Bars`, which contain the open, high, low and close prices for a given time period.

```{code-cell} python
feed = rq.feeds.YahooFeed("TSLA", "MSFT", "GOOG", start_date="2010-01-10")
```

When we have the feed we can plot the data to see what it looks like. The `plot` method takes a symbol as an argument and will plot the price of that symbol over time. It can help to get a better understanding of the data we are working with and detect any anomalies or outliers in the data.

```{code-cell} python
feed.plot("TSLA");
```

## Strategy
A strategy is the core of any back test. It defines the rules for when to buy and sell an asset. 
In this example we use a Exponential Moving Average Crossover strategy, which is included in *roboquant* out of the box.

But normally you would create your own strategy by subclassing the `Strategy` class and implementing the `generate_signals` method. This method creates signals with ratings based on the data provided by the feed. These signals are then used to create orders by a `Trader`, which are executed by the `Broker`.


```{code-cell} python
strategy = rq.strategies.EMACrossover()
```

## Run
Now we can run the back test using the feed and strategy we just created. The `run` function takes many different parameters, making it suitable from back testing all the way to live trading. In this example we only provide the `feed` and `strategy` and leave the other parameters to their default values.   

The result of the back test is an `Account` object, which contains all the trades that were executed during the back test and various other trading account related information.

```{code-cell} python
account = rq.run(feed, strategy)
print(account)
```

We can now also plot the trades that were executed during the back test. This is done by calling the `plot` method on the feed, and passing in the symbol of the asset we want to plot, as well as the trades that were executed.

```{code-cell} python
feed.plot("TSLA", trades = account.trades);
```

## Next steps
This page showed how to run a simple back test using *roboquant*. For each of the core components of a back test, there are more advanced features available. 

For example, you can create your own strategy, use a different feed provider, or use a different broker. You can also run a back test on multiple asset classes at the same time, or use a different time frame for the back test.