import pandas as pd
import numpy as np
from dataclasses import dataclass

from .StockInfoCache import StockInfoCache
from .Fundamentus_Page import Fundamentus_Page


@dataclass
class DividendReader:
    br_tickers: list
    us_tickers: list
    fii_tickers: list
    start_date: str = "2018-01-01"
    cache_file: str = "debug/cache_dividends.tsv"

    def __post_init__(self):
        self.df = pd.DataFrame(columns=["SYMBOL", "PRICE", "PAYDATE", "OPERATION"])
        self.cache = StockInfoCache(self.cache_file)

    def load(self):
        if not self.cache.is_updated():
            if self.br_tickers != None and len(self.br_tickers) > 0:
                self.df = self.loadData(self.br_tickers, type="ação")

            if self.fii_tickers != None and len(self.fii_tickers) > 0:
                tmp = self.loadData(self.fii_tickers, "fii")
                self.df = tmp if self.df.empty else pd.concat([self.df, tmp])

            if self.us_tickers != None and len(self.us_tickers) > 0:
                tmp = self.loadData(self.us_tickers, "stock")
                self.df = tmp if self.df.empty else pd.concat([self.df, tmp])

            self.df = self.cache.merge(self.df, sortby=["DATE", "SYMBOL"], on=["SYMBOL", "DATE", "OPERATION"])
        else:
            self.df = self.cache.load_data()
            self.df["PAYDATE"] = pd.to_datetime(self.df["PAYDATE"], format="%Y-%m-%d")

        if not self.df.empty:
            self.df.set_index("DATE", inplace=True)
            self.df["PRICE"] -= self.df["TAX"]
            self.df["OPERATION"] = self.df["OPERATION"].map(lambda x: "D" if x == "JCP" else x)
            self.df = self.df[["SYMBOL", "PRICE", "PAYDATE", "OPERATION"]]

    def loadData(self, paperList, type):
        tb = pd.DataFrame()
        # pageObj = ADVFN_Page()
        pageObj = Fundamentus_Page(type)

        for paper in paperList:
            rawTable = pageObj.read(paper)
            if rawTable.empty:
                continue

            # print(rawTable)
            rawTable["SYMBOL"] = paper

            # Discount a tax of 15% when is JCP (Juros sobre capital proprio)
            rawTable["TAX"] = np.where(rawTable["OPERATION"] == "JCP", rawTable["PRICE"] * 0.15, 0)

            rawTable["PAYDATE"] = np.where(rawTable["PAYDATE"] == "-", rawTable["DATE"], rawTable["PAYDATE"])
            rawTable["PAYDATE"] = pd.to_datetime(rawTable["PAYDATE"], format="%d-%m-%Y")
            rawTable["DATE"] = pd.to_datetime(rawTable["DATE"], format="%d-%m-%Y")
            rawTable = rawTable[["SYMBOL", "DATE", "PRICE", "PAYDATE", "OPERATION", "TAX"]]

            tb = pd.concat([tb, rawTable])
        # print(tb)
        return tb[tb["DATE"] >= self.start_date]

    def getPeriod(self, paper, fromDate, toDate):
        filtered = self.df[self.df["SYMBOL"] == paper].loc[fromDate:toDate]
        return filtered[["SYMBOL", "PRICE", "PAYDATE", "OPERATION"]]