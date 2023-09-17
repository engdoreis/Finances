import pandas as pd
import yfinance as yf

from .StockInfoCache import StockInfoCache

from data import DataSchema


class SplitsReader:
    def __init__(self, brTickers, usTickers, start_date="2018-01-01", cache="debug/cache_splits.tsv"):
        self.br_tickers = [t + ".SA" for t in brTickers]
        self.us_tickers = usTickers if usTickers is not None else []
        self.start_date = start_date
        self.df = pd.DataFrame()
        self.cache = StockInfoCache(cache)

    def load(self):
        if not self.cache.is_updated():
            if len(self.br_tickers) > 0:
                self.df = pd.concat([self.df, self.loadData(self.br_tickers)])

            if len(self.us_tickers) > 0:
                self.df = pd.concat([self.df, self.loadData(self.us_tickers)])

            self.df = self.cache.merge(self.df)
        else:
            self.df = self.cache.load_data()

        self.df.set_index(DataSchema.DATE, inplace=True)

    def getPeriod(self, ticker, fromDate, toDate):
        filtered = self.df[self.df[DataSchema.SYMBOL] == ticker].loc[fromDate:toDate]
        return filtered[[DataSchema.SYMBOL, DataSchema.QTY]]

    def get_accumulated(self, ticker, start_date):
        filtered = self.df[self.df[DataSchema.SYMBOL] == ticker].loc[start_date:]
        return filtered[DataSchema.QTY].prod() if len(filtered) > 0 else 1

    def loadData(self, tickerList):
        res = pd.DataFrame()
        for ticker in tickerList:
            try:
                data = pd.DataFrame(yf.Ticker(ticker).splits)
            except:
                continue
            data[DataSchema.SYMBOL] = ticker.replace(".SA", "")
            res = pd.concat([res, data], axis=0)
        res.index.rename(DataSchema.DATE, inplace=True)
        res.columns = [DataSchema.SYMBOL, DataSchema.QTY]
        res = res.reset_index()
        res[DataSchema.DATE] = pd.to_datetime(res[DataSchema.DATE], format="%Y-%m-%d").dt.tz_localize(None)
        return res[res[DataSchema.DATE] > self.start_date]
