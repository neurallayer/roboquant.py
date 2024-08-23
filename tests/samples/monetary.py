# %%
from roboquant.monetary import EUR, USD, JPY, ECBConversion
from datetime import datetime

# %%
wallet = 20@EUR + 10@USD + 1000@JPY + 10@USD
print("The wallet contains", wallet)

# %%
ECBConversion().register()
print("The wallet is", wallet@EUR)
print("The wallet is", wallet@USD)

print("100@USD =", 100@USD@JPY)
# %%
dt1 = datetime.fromisoformat("2010-01-01-00:00:00+00:00")
print("Value of wallet in USD in 2010 is", wallet.convert_to(USD, dt1))

dt2 = datetime.fromisoformat("2020-01-01-00:00:00+00:00")
print("Value of wallet in USD in 2020 is", wallet.convert_to(USD, dt2))
