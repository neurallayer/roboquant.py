# %%
from datetime import datetime
from roboquant.monetary import EUR, USD, JPY, ECBConversion, Amount

# %%
# Different ways to create Amounts and add them to a wallet
wallet = 20@EUR + 10@USD + 1000@JPY + 10@USD
wallet += EUR(10.0)
wallet += Amount(EUR, 10)
print("The wallet contains", wallet)

# %%
# Install the ECB currency converter
ECBConversion().register()

# Convert a wallet to a single currency
print("The total value of the wallet today is", wallet@EUR)
print("The total value of the wallet today is", wallet@USD)

# Convert amounts
print("100@USD =", 100@USD@JPY)
# %%
dt1 = datetime.fromisoformat("2010-01-01-00:00:00+00:00")
print("Value of wallet in USD in 2010 is", wallet.convert_to(USD, dt1))

dt2 = datetime.fromisoformat("2020-01-01-00:00:00+00:00")
print("Value of wallet in USD in 2020 is", wallet.convert_to(USD, dt2))
