# %%
from time import sleep
import logging
from roboquant import Order
from roboquant.ibkr.broker import IBKRBroker

# %%
logging.basicConfig()
logging.getLogger("roboquant").setLevel(level=logging.INFO)

ibkr = IBKRBroker.use_gateway()
account = ibkr.sync()
print(account)

orders = [Order(asset, -pos.size, pos.mkt_price) for asset, pos in account.positions.items()]

# close all positions
ibkr.place_orders(orders)
for _ in range(20):
    sleep(1)
    account = ibkr.sync()
    print()
    print(account)
    if not account.orders:
        break

ibkr.disconnect()
