from time import sleep
import logging
from roboquant import Order
from roboquant.brokers.ibkrbroker import IBKRBroker
import sys

if __name__ == "__main__":
    logging.basicConfig()
    logging.getLogger("roboquant").setLevel(level=logging.INFO)

    ibkr = IBKRBroker()
    account = ibkr.sync()
    print(account)

    orders = [Order(symbol, - pos.size) for symbol, pos in account.positions.items()]
    ibkr.place_orders(orders)
    for _ in range(60):
        sleep(3)
        account = ibkr.sync()
        print()
        print(account)

    sys.exit(0)
