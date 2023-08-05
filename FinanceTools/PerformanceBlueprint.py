import pandas as pd

from .Portifolio import Portifolio


class PerformanceBlueprint:
    def __init__(self, priceReader, dataframe, date, currency="R$"):
        self.currency = currency
        self.pcRdr = priceReader
        self.equity = (
            self.cost
        ) = (
            self.realizedProfit
        ) = (
            self.div
        ) = self.paperProfit = self.profit = self.usdIbov = self.ibov = self.sp500 = self.profitRate = self.expense = 0
        self.date = date
        self.df = dataframe[(dataframe["DATE"] <= date)].copy(deep=True)
        if not self.df.empty:
            priceReader.setFillDate(self.date)
            self.pt = Portifolio(self.pcRdr, self.df)

    def calc(self):
        if not self.df.empty:
            ptf = self.pt.dtframe
            self.equity = (ptf["PRICE"] * ptf["QUANTITY"]).sum()
            self.cost = ptf["COST"].sum()
            self.realizedProfit = self.df.loc[self.df.OPERATION == "S", "Profit"].sum()
            self.div = self.df[self.df.OPERATION.isin(["D1", "A1", "R1", "JCP1", "D", "A", "R", "JCP", "CF"])][
                "AMOUNT"
            ].sum()
            self.paperProfit = self.equity - self.cost
            self.profit = self.equity - self.cost + self.realizedProfit + self.div
            self.profitRate = self.profit / self.cost
            indexHistory = self.pcRdr.getIndexHistory("IBOV", self.date)
            self.ibov = indexHistory.iloc[-1] / indexHistory.iloc[0] - 1
            indexHistory = self.pcRdr.getIndexHistory("S&P500", self.date)
            self.sp500 = indexHistory.iloc[-1] / indexHistory.iloc[0] - 1

            indexHistory = self.pcRdr.getIndexHistory("selic", self.date)
            self.selic = indexHistory.iloc[-1]
            self.cum_cdb = indexHistory.apply(lambda y: ((y + 1) ** (1 / 365))).cumprod().iloc[-1] - 1

            self.expense = self.df.loc[self.df.OPERATION == "B", "FEE"].sum()
            self.exchangeRatio = self.pcRdr.getIndexCurrentValue("USD", self.date)
            return self
