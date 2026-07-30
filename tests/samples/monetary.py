# %% [markdown]
# This example shows how to use the `roboquant` library to handle monetary amounts and wallets.
# It demonstrates how to create amounts in different currencies, manage wallets, and convert between currencies using
# the European Central Bank (ECB) conversion rates.

# %%
from datetime import datetime, timedelta
from roboquant import utcnow
from roboquant.common.monetary import EUR, USD, JPY, ECBConversion, Amount, Wallet

# %%
# Different ways to create an amount.
amt1 = 20@USD
amt2 = USD(20)
amt3 = Amount(USD, 20.0)
assert amt1 == amt2 == amt3

# %%
# A wallet can contain amounts of different currencies. A wallet
# is also mutable. It behaves very much like a `dict[Currency, float]`
#
# The are different ways to create a new instance of `Wallet`
wallet1 = 20@EUR + 10@USD + 1000@JPY + 10@USD
wallet2 = Wallet(20@EUR, 10@USD, 1000@JPY, 10@USD)
assert wallet1 == wallet2
assert wallet1[JPY] == 1000

#%% [markdown]
# It might come as bit of a suprise, but adding two amounts will
# always return a `Wallet`, even if the amounts are denoted
# in the same currency.
wallet3 = 20@EUR + 30@EUR
assert isinstance(wallet3, Wallet)

#%% [mrkdown]
# There are several ways you can add or subtract amounts
# from an existing wallet.
wallet1 += EUR(10.0)
wallet1 += Amount(EUR, 10)
wallet1 -= 50@USD
wallet1.deposit(10@EUR)
wallet1.withdraw(20@EUR)
assert wallet1[EUR] == 30.0
print("The wallet contains", wallet1)

# %% [markdown]
# Roboquant supports trading of assets in different currencies.
# But for this to work correctly, a currency converter needs to registered.
#
# By default the `NoConverter` is registered that will raise an Exception if a
# conversion is actually required.
#
# Roboquant comes with a European Central Bank conversion that can be
# used if you only need support for fiat currencies. It contains the daily
# exchange rates between all major currencies since the year 2000.
#
# Upon instantiation it will download the latest exchange rates from the ECB
# website.


# %%
ECBConversion().register()

# %% [markdown]
# Now we can convert between currencies.
# For example convert a wallet to a single currency at todays exchange rate:
print("The total value of the wallet today is", wallet1@EUR)
print("The total value of the wallet today is", wallet1@USD)

# %%
# Or convert an amount from one currency to another one:
amt = 100@USD
print(amt, "=", amt@JPY)


# %% [markdown]
# But it is also possible to convert an amount or wallet against an
# exchage rate in the past
#%%
yesterday = utcnow() - timedelta(days=1)
print(amt,"in JYP yesterday is", amt.convert_to(JPY, yesterday))

# %%
# Convert a wallet to a single currency at different dates
dt1 = datetime.fromisoformat("2010-01-01")
print("Value of wallet in USD in 2010 is", wallet1.convert_to(USD, dt1))

dt2 = datetime.fromisoformat("2020-01-01")
print("Value of wallet in USD in 2020 is", wallet1.convert_to(USD, dt2))
