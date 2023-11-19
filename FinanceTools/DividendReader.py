import pandas as pd
import numpy as np
from dataclasses import dataclass

from .StockInfoCache import StockInfoCache
from .Fundamentus_Page import Fundamentus_Page

from data import DataSchema


@dataclass
class DividendReader:
    br_tickers: list
    us_tickers: list
    fii_tickers: list
    start_date: str = "2018-01-01"
    cache_file: str = "debug/cache_dividends.tsv"

    def __post_init__(self):
        self.df = pd.DataFrame(columns=[DataSchema.SYMBOL, DataSchema.PRICE, DataSchema.PAYDATE, DataSchema.OPERATION])
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

            self.df = self.cache.merge(
                self.df,
                sortby=[DataSchema.DATE, DataSchema.SYMBOL],
                on=[DataSchema.SYMBOL, DataSchema.DATE, DataSchema.OPERATION],
            )
        else:
            self.df = self.cache.load_data()
            self.df[DataSchema.PAYDATE] = pd.to_datetime(self.df[DataSchema.PAYDATE], format=DataSchema.DATE_FORMAT)

        if not self.df.empty:
            self.df.set_index(DataSchema.DATE, inplace=True)
            self.df[DataSchema.PRICE] -= self.df["TAX"]
            self.df[DataSchema.OPERATION] = self.df[DataSchema.OPERATION].map(lambda x: "D" if x == "JCP" else x)
            self.df = self.df[[DataSchema.SYMBOL, DataSchema.PRICE, DataSchema.PAYDATE, DataSchema.OPERATION]]

    def loadData(self, paperList, type):
        tb = pd.DataFrame()
        # pageObj = ADVFN_Page()
        pageObj = Fundamentus_Page(type)

        for paper in paperList:
            rawTable = pageObj.read(paper)
            if rawTable.empty:
                continue

            # print(rawTable)
            rawTable[DataSchema.SYMBOL] = paper

            # Discount a tax of 15% when is JCP (Juros sobre capital proprio)
            rawTable["TAX"] = np.where(rawTable[DataSchema.OPERATION] == "JCP", rawTable[DataSchema.PRICE] * 0.15, 0)

            rawTable[DataSchema.PAYDATE] = np.where(
                rawTable[DataSchema.PAYDATE] == "-", rawTable[DataSchema.DATE], rawTable[DataSchema.PAYDATE]
            )
            rawTable[DataSchema.PAYDATE] = pd.to_datetime(rawTable[DataSchema.PAYDATE], format=DataSchema.DATE_FORMAT)
            rawTable[DataSchema.DATE] = pd.to_datetime(rawTable[DataSchema.DATE], format=DataSchema.DATE_FORMAT)
            rawTable = rawTable[
                [DataSchema.SYMBOL, DataSchema.DATE, DataSchema.PRICE, DataSchema.PAYDATE, DataSchema.OPERATION, "TAX"]
            ]

            tb = pd.concat([tb, rawTable])
        # print(tb)
        return tb[tb[DataSchema.DATE] >= self.start_date]

    def getPeriod(self, paper, fromDate, toDate):
        filtered = self.df[self.df[DataSchema.SYMBOL] == paper].loc[fromDate:toDate]
        return filtered[[DataSchema.SYMBOL, DataSchema.PRICE, DataSchema.PAYDATE, DataSchema.OPERATION]]
