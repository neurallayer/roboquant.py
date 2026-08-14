# %%
from time import sleep
import logging
import roboquant as rq
from roboquant.brokers.ibkr import IBKRBroker

# %%
logging.basicConfig()
logging.getLogger("roboquant").setLevel(level=logging.INFO)

ibkr = IBKRBroker()
account = ibkr.sync()
print(account)

# close all positions if there is not already an open order for that same asset
open_order_assets = {order.asset for order in account.orders}
orders = [
    rq.Order.market_order(asset, -pos.size, pos.mkt_price)
    for asset, pos in account.portfolio.items()
    if asset not in open_order_assets
]

ibkr.place_orders(orders)
for _ in range(10):
    sleep(5)
    account = ibkr.sync()
    print(account)
    if not account.orders:
        break
