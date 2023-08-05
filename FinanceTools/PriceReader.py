import datetime as dt
from dataclasses import dataclass

import numpy as np
import pandas as pd
import yfinance as yf
from bcb import sgs


@dataclass
class PriceReader:
    br_tickers: list
    us_tickers: list
    start_date: str = "2018-01-01"

    def __post_init__(self):
        self.start_date = self.start_date.strftime("%Y-%m-%d")
        self.fillDate = dt.datetime.today().strftime("%m-%d-%Y")
        self.df = pd.DataFrame(columns=["Date"])

    def load(self):
        # Read BR market data
        if (self.br_tickers != None) and (len(self.br_tickers) > 0):
            self.df = self.readData(self.br_tickers, self.start_date).reset_index()
            self.df.columns = self.df.columns.str.removesuffix(".SA")

        # Read US Market data
        if (self.us_tickers != None) and (len(self.us_tickers) > 0):
            self.df = self.df.merge(
                self.readUSData(self.us_tickers, self.start_date).reset_index(), how="outer", on="Date"
            )

        self.df = self.df.set_index("Date").sort_index()
        # self.df.to_csv('debug.csv', sep='\t')

        indexList = ["^BVSP", "^GSPC", "BRLUSD=X"]
        self.brlIndex = self.readUSData(indexList, self.start_date).reset_index()
        self.brlIndex.rename(columns={"^BVSP": "IBOV", "^GSPC": "S&P500", "BRLUSD=X": "USD"}, inplace=True)
        self.brlIndex = self.brlIndex.merge(self.read_br_selic(self.start_date), on="Date")
        self.brlIndex = self.brlIndex.set_index("Date")

    def setFillDate(self, date):
        self.fillDate = date

    def fillCurrentValue(self, row):
        row["PRICE"] = self.getCurrentValue(row["SYMBOL"], self.fillDate)
        return row

    def readData(self, code, start_date="2018-01-01"):
        s = ""
        for c in code:
            s += c + ".SA "

        tks = yf.Tickers(s)
        dfs = tks.history(start=start_date, timeout=1000)[["Close"]]
        dfs.columns = dfs.columns.droplevel()
        return dfs

    def readUSData(self, code, start_date="2018-01-01"):
        s = ""
        for c in code:
            s += c + " "

        tks = yf.Tickers(s)
        dfs = tks.history(start=start_date)[["Close"]]
        dfs.columns = dfs.columns.droplevel()
        return dfs

    def read_br_selic(self, start_date="2018-01-01"):
        try:
            selic = sgs.get({"selic": 432}, start=start_date)
            selic["selic"] /= 100
            return selic
        except:
            return pd.DataFrame(columns=["Date", "selic"])

    def getHistory(self, code, start="2018-01-01"):
        return self.df.loc[start:][code]

    def getCurrentValue(self, code, date=None):
        if not code in self.df:
            return np.nan

        if date == None:
            return self.df.iloc[-1][code]

        available, date = self.checkLastAvailable(self.df, date, code)
        if available:
            return self.df.loc[date][code]
        return self.df.iloc[0][code]

    def getIndexHistory(self, code, end):
        ret = self.brlIndex.loc[:end][code]
        return ret.dropna()

    def getIndexCurrentValue(self, code, date=None):
        if len(self.brlIndex) == 0:
            return 1

        if date == None:
            return self.brlIndex.iloc[-1][code]

        available, date = self.checkLastAvailable(self.brlIndex, date, code)
        if available:
            return self.brlIndex.loc[date][code]
        return self.brlIndex.iloc[0][code]

    def checkLastAvailable(self, dtframe, loockDate, field):
        date = pd.to_datetime(loockDate)
        day = pd.Timedelta(1, unit="d")
        # Look for last available date

        while (not (date in dtframe.index)) or pd.isna(dtframe.loc[date][field]):
            date = date - day
            if date < dtframe.index[0]:
                return False, 0
        return True, date
