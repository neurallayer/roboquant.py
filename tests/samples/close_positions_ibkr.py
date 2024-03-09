from time import sleep
import logging
from roboquant import Order
from roboquant.brokers.ibkrbroker import IBKRBroker


if __name__ == "__main__":
    logging.basicConfig()
    logging.getLogger("roboquant").setLevel(level=logging.INFO)

    ibkr = IBKRBroker.use_tws()
    account = ibkr.sync()
    print(account)

    orders = [Order(symbol, - pos.size) for symbol, pos in account.positions.items()]

    # close all but the first 10 positions
    ibkr.place_orders(orders[10:])
    for _ in range(20):
        sleep(1)
        account = ibkr.sync()
        print()
        print(account)

    ibkr.disconnect()
