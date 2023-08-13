import numpy as np
import pandas as pd

from .PerformanceBlueprint import PerformanceBlueprint
from .Color import Color


class PerformanceViewer:
    def __init__(self, *args):

        if len(args) == 2 and isinstance(args[0], pd.DataFrame):
            row = args[0].set_index("Date").loc[args[1]]
            self.buildTable(
                row["Equity"],
                row["Cost"],
                row["Expense"],
                row["paperProfit"],
                row["Profit"],
                row["Div"],
                row["TotalProfit"],
                row["selic"],
                row["Ibov"],
                row["SP500"],
            )
        elif isinstance(args[0], PerformanceBlueprint):
            p = args[0]
            self.buildTable(
                p.equity,
                p.cost,
                p.expense,
                p.paperProfit,
                p.realizedProfit,
                p.div,
                p.profit,
                p.cum_cdb,
                p.ibov,
                p.sp500,
                p.currency,
                p.exchangeRatio,
            )

    def buildTable(
        self,
        equity,
        cost,
        expense,
        paperProfit,
        profit,
        div,
        totalProfit,
        selic,
        ibov,
        sp500,
        currency="USD",
        exchangeRatio=0.22,
    ):
        self.pf = pd.DataFrame(columns=["Item", currency])
        self.pf.loc[len(self.pf)] = ["Equity          ", equity]
        self.pf.loc[len(self.pf)] = ["Cost            ", cost]
        self.pf.loc[len(self.pf)] = ["Expenses        ", expense]
        self.pf.loc[len(self.pf)] = ["Paper profit    ", paperProfit]
        self.pf.loc[len(self.pf)] = ["Realized profit ", profit]
        self.pf.loc[len(self.pf)] = ["Dividends       ", div]
        self.pf.loc[len(self.pf)] = ["Total Profit    ", totalProfit]

        self.pf["%"] = self.pf[currency] / cost

        self.pf.loc[len(self.pf)] = ["Selic    ", 0, selic]
        self.pf.loc[len(self.pf)] = ["Ibov     ", 0, ibov]
        self.pf.loc[len(self.pf)] = ["S&P500   ", 0, sp500]
        self.pf.loc[:, "%"] *= 100

        if currency != "USD":
            self.pf["USD"] =  self.pf[currency] * exchangeRatio

        self.pf.set_index("Item", inplace=True)

    def show(self):
        format_dict = {"USD": " {:^,.2f}", "BRL": " {:^,.2f}", "GBP": " {:^,.2f}", "%": " {:>.1f}%"}
        return self.pf.style.applymap(Color().color_negative_red).format(format_dict)
