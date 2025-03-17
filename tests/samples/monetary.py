# %%
from datetime import datetime, timedelta
from roboquant import utcnow
from roboquant.monetary import EUR, USD, JPY, ECBConversion, Amount, Wallet

# %%
# Different ways to create Amounts
amt1 = 20@USD
amt2 = USD(20)
amt3 = Amount(USD, 20)
assert amt1 == amt2 == amt3

# %%
# Different ways to create and modify wallets
wallet = 20@EUR + 10@USD + 1000@JPY + 10@USD
assert isinstance(wallet, Wallet)
assert wallet == Wallet(*wallet.amounts())

wallet += EUR(10.0)
wallet += Amount(EUR, 10)
assert wallet[EUR] == 40.0
print("The wallet contains", wallet)

# %%
# Install the ECB currency converter
ECBConversion().register()

# %%
# Convert a wallet to a single currency at todays exchange rate
print("The total value of the wallet today is", wallet@EUR)
print("The total value of the wallet today is", wallet@USD)

# %%
# Convert between amounts
amt = 100@USD
print("100@USD =", amt@JPY)

yesterday = utcnow() - timedelta(days=1)
print("100@@USD", amt.convert_to(JPY, yesterday))

# %%
# Convert a wallet to a single currency at different dates
dt1 = datetime.fromisoformat("2010-01-01")
print("Value of wallet in USD in 2010 is", wallet.convert_to(USD, dt1))

dt2 = datetime.fromisoformat("2020-01-01")
print("Value of wallet in USD in 2020 is", wallet.convert_to(USD, dt2))

# %%
