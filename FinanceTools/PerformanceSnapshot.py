import pandas as pd

from .Portfolio import Portfolio

from data import DataSchema


class PerformanceSnapshot:
    equity = cost = realizedProfit = div = paperProfit = profit = usdIbov = ibov = sp500 = profitRate = expense = 0

    def __init__(self, price_reader, split_reader, dataframe, date, currency="USD"):
        self.currency = currency
        self.price_reader = price_reader
        self.date = date
        self.df = dataframe[(dataframe[DataSchema.DATE] <= date)].copy(deep=True)
        if not self.df.empty:
            self.portfolio = Portfolio(self.price_reader, split_reader, date, self.df)

    def calc(self):
        if not self.df.empty:
            ptf = self.portfolio.dtframe
            self.equity = (ptf[DataSchema.PRICE] * ptf[DataSchema.QTY]).sum()
            self.cost = ptf["COST"].sum()
            self.realizedProfit = self.df.loc[self.df.OPERATION == "S", DataSchema.PROFIT].sum()
            self.div = self.df[self.df.OPERATION.isin(["D1", "T1", "A1", "R1", "JCP1", "D", "A", "R", "JCP", "CF"])][
                DataSchema.AMOUNT
            ].sum()
            self.paperProfit = self.equity - self.cost
            self.profit = self.equity - self.cost + self.realizedProfit + self.div
            self.profitRate = self.profit / self.cost
            indexHistory = self.price_reader.getIndexHistory("IBOV", self.date)
            self.ibov = indexHistory.iloc[-1] / indexHistory.iloc[0] - 1
            indexHistory = self.price_reader.getIndexHistory("S&P500", self.date)
            self.sp500 = indexHistory.iloc[-1] / indexHistory.iloc[0] - 1

            indexHistory = self.price_reader.getIndexHistory("selic", self.date)
            self.selic = indexHistory.iloc[-1]
            self.cum_cdb = indexHistory.apply(lambda y: ((y + 1) ** (1 / 365))).cumprod().iloc[-1] - 1

            self.expense = self.df.loc[self.df.OPERATION == "B", DataSchema.FEES].sum()
            self.exchangeRatio = (
                1
                if self.currency == "USD"
                else self.price_reader.getIndexCurrentValue(self.currency + "USD", self.date)
            )
            return self
