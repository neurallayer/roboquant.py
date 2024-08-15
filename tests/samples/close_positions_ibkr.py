# %%
from time import sleep
import logging
from roboquant import Order
from roboquant.brokers.ibkr import IBKRBroker

# %%
logging.basicConfig()
logging.getLogger("roboquant").setLevel(level=logging.INFO)

ibkr = IBKRBroker.use_tws()
account = ibkr.sync()
print(account)

orders = [Order(asset, -pos.size, pos.mkt_price) for asset, pos in account.positions.items()]

# close all but the first 10 positions
ibkr.place_orders(orders[10:])
for _ in range(20):
    sleep(1)
    account = ibkr.sync()
    print()
    print(account)

ibkr.disconnect()
