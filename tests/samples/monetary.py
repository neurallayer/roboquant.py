# %%
from datetime import datetime, timedelta, timezone
from roboquant.monetary import EUR, USD, JPY, ECBConversion, Amount, Wallet

# %%
amt1 = 20@USD
amt2 = amt1 + 30@USD
assert isinstance(amt2, Amount)
assert amt2 == 50@USD

# %%
# Different ways to create Amounts and add them to a wallet
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

yesterday = datetime.now(timezone.utc) - timedelta(days=1)
print("100@@USD", amt.convert_to(JPY, yesterday))

# %%
# Convert a wallet to a single currency at different dates
dt1 = datetime.fromisoformat("2010-01-01")
print("Value of wallet in USD in 2010 is", wallet.convert_to(USD, dt1))

dt2 = datetime.fromisoformat("2020-01-01")
print("Value of wallet in USD in 2020 is", wallet.convert_to(USD, dt2))
