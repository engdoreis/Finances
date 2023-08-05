import pandas as pd
import numpy as np
import datetime as dt
import yfinance as yf


class PriceReader:
    def __init__(self, brTickerList, usTickerList, startDate="2018-01-01"):
        self.brTickerList = brTickerList
        self.usTickerList = usTickerList
        self.startDate = startDate.strftime("%Y-%m-%d")
        self.fillDate = dt.datetime.today().strftime("%m-%d-%Y")
        self.df = pd.DataFrame(columns=["Date"])

    def load(self):
        # Read BR market data
        if (self.brTickerList != None) and (len(self.brTickerList) > 0):
            self.df = self.readData(self.brTickerList, self.startDate).reset_index()
            self.df.columns = self.df.columns.str.removesuffix(".SA")

        # Read US Market data
        if (self.usTickerList != None) and (len(self.usTickerList) > 0):
            self.df = self.df.merge(
                self.readUSData(self.usTickerList, self.startDate).reset_index(), how="outer", on="Date"
            )

        self.df = self.df.set_index("Date").sort_index()
        # self.df.to_csv('debug.csv', sep='\t')

        indexList = ["^BVSP", "^GSPC", "BRLUSD=X"]
        self.brlIndex = self.readUSData(indexList, self.startDate).reset_index()
        self.brlIndex.rename(columns={"^BVSP": "IBOV", "^GSPC": "S&P500", "BRLUSD=X": "USD"}, inplace=True)
        self.brlIndex = self.brlIndex.merge(self.read_br_selic(self.startDate), on="Date")
        self.brlIndex = self.brlIndex.set_index("Date")

    def setFillDate(self, date):
        self.fillDate = date

    def fillCurrentValue(self, row):
        row["PRICE"] = self.getCurrentValue(row["SYMBOL"], self.fillDate)
        return row

    def readData(self, code, startDate="2018-01-01"):
        s = ""
        for c in code:
            s += c + ".SA "

        tks = yf.Tickers(s)
        dfs = tks.history(start=startDate, timeout=1000)[["Close"]]
        dfs.columns = dfs.columns.droplevel()
        return dfs

    def readUSData(self, code, startDate="2018-01-01"):
        s = ""
        for c in code:
            s += c + " "

        tks = yf.Tickers(s)
        dfs = tks.history(start=startDate)[["Close"]]
        dfs.columns = dfs.columns.droplevel()
        return dfs

    def read_br_selic(self, startDate="2018-01-01"):
        from bcb import sgs

        try:
            selic = sgs.get({"selic": 432}, start=startDate)
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
