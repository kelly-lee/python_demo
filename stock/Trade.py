from ib.opt import Connection, message
from ib.ext.Contract import Contract
from ib.ext.Order import Order


def operate(symbol, action, quantity, price=None):
    contract = Contract()
    contract.m_symbol = symbol
    contract.m_secType = 'STK'
    contract.m_exchange = 'SMART'
    contract.m_primaryExch = 'ISLAND'
    contract.m_currency = 'USD'

    order = Order()
    if price is not None:
        order.m_orderType = 'LMT'
        order.m_lmtPrice = price
    else:
        order.m_orderType = 'MKT'

    order.m_totalQuantity = quantity
    order.m_action = action

    ibConnection.placeOrder(12, contract, order)


ibConnection = None

ibConnection = Connection.create(port=7497, clientId=1314)
ibConnection.connect()
operate(symbol='AAPL', action='BUY', quantity=1, price=156.7)
ibConnection.disconnect()
