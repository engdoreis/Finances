import json
from datetime import datetime
import pandas as pd
from collections import namedtuple
from FinanceTools import *
import numpy as np
import sys

class Simulator:
    def __init__(self, tickers, value, startDate):
        self.prcReader = PriceReader(tickers, startDate)
        self.prcReader.load()
        self.value = value
        self.orders = []
        self.restValue = {}
        self.startDate = startDate

    def ProcessTickers(self, stocks, date):
        order = namedtuple('order', 'Ticker Date Value Qty Type Category Fee')
        for stock in stocks:
            entryDate = stock['EntryDate']
            if ( pd.to_datetime(entryDate) >  pd.to_datetime(date)): continue

            part = float(stock['Participation'])
            ticker = stock['Ticker']
            price = self.prcReader.getCurrentValue(ticker, date)
            if (price == np.nan): continue

            availableValue = ((part * self.value) + self.restValue.get(ticker, 0))
            qty =  availableValue // price
            if(qty > 0):
                self.orders.append(order(ticker, date, price, qty, 'Compra', stock['Category'], (price*qty*0.0003)))
            rest = availableValue % price
            if rest > 0:
                self.restValue.update({ticker: rest})

    def ProcessTimeline(self, stocks):
        monthList = pd.date_range(start=self.startDate, end=datetime.today(), freq='MS').format(formatter=lambda x: x.strftime('%Y-%m-%d'))
        for date in monthList:
            self.ProcessTickers(stocks, date)

    def Dataframe(self):
        return pd.DataFrame(self.orders)

def GenerateWallet(inFile='simulator.json', outFile='orders.csv'):
    with open(inFile) as file:
        setup = json.load(file)

    value = float(setup['BuyValue'])

    tickers = []
    for stock in setup['stocks']:
        tickers.append(stock['Ticker'])

    sim = Simulator(tickers, value, setup['StartDate'])
    sim.ProcessTimeline(setup['stocks'])
    sim.Dataframe().to_csv(outFile, index=False)

if __name__ == "__main__":
    name = str(sys.argv[1])
    GenerateWallet(name + '.json', name + '.csv')
