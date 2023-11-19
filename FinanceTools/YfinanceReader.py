import pandas as pd
import yfinance as yf
from .DividendReader import DividendReader

from data import DataSchema

# 30% tax
tax_rate = 0.3


class YfinanceReader(DividendReader):
    def loadData(self, paperList, type=None):
        res = pd.DataFrame()

        for paper in paperList:
            try:
                data = pd.DataFrame(yf.Ticker(paper).dividends)
            except:
                continue
            if not data.empty:
                data[DataSchema.SYMBOL] = paper.replace(".SA", "")
                res = pd.concat([res, data], axis=0)

        res.reset_index(inplace=True)
        res.rename(columns={"Date": DataSchema.DATE, "Dividends": DataSchema.PRICE}, inplace=True)
        res[DataSchema.PAYDATE] = res[DataSchema.DATE] = pd.to_datetime(
            res[DataSchema.DATE], format=DataSchema.DATE_FORMAT
        ).dt.tz_localize(None)
        res = res[res[DataSchema.DATE] >= self.start_date]
        res["TAX"] = res[DataSchema.PRICE] * tax_rate * 0
        res = res[[DataSchema.SYMBOL, DataSchema.DATE, DataSchema.PRICE, DataSchema.PAYDATE, "TAX"]]
        res[DataSchema.OPERATION] = "D2"
        return res
