import pandas as pd

from .StockInfoCache import StockInfoCache


class SplitsReader:
    def __init__(self, brTickers, usTickers, startDate="2018-01-01", cache="debug/cache_splits.tsv"):
        self.brTickerList = [t + ".SA" for t in brTickers]
        self.usTickerList = usTickers if usTickers is not None else []
        self.startDate = startDate
        self.df = pd.DataFrame()
        self.cache = StockInfoCache(cache)

    def load(self):
        if not self.cache.is_updated():
            if len(self.brTickerList) > 0:
                self.df = pd.concat([self.df, self.loadData(self.brTickerList)])

            if len(self.usTickerList) > 0:
                self.df = pd.concat([self.df, self.loadData(self.usTickerList)])

            self.df = self.cache.merge(self.df)
        else:
            self.df = self.cache.load_data()

        self.df.set_index("DATE", inplace=True)

    def getPeriod(self, ticker, fromDate, toDate):
        filtered = self.df[self.df["SYMBOL"] == ticker].loc[fromDate:toDate]
        return filtered[["SYMBOL", "QUANTITY"]]

    def loadData(self, tickerList):
        res = pd.DataFrame()
        for ticker in tickerList:
            try:
                data = pd.DataFrame(yf.Ticker(ticker).splits)
            except:
                continue
            data["SYMBOL"] = ticker.replace(".SA", "")
            res = pd.concat([res, data], axis=0)
        res.index.rename("DATE", inplace=True)
        res.columns = ["SYMBOL", "QUANTITY"]
        res = res.reset_index()
        res["DATE"] = pd.to_datetime(res["DATE"], format="%Y-%m-%d").dt.tz_localize(None)
        return res[res["DATE"] > self.startDate]
