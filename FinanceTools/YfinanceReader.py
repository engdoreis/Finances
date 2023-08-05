import pandas as pd
import yfinance as yf
from .DividendReader import DividendReader


class YfinanceReader(DividendReader):
    def loadData(self, paperList, type=None):
        res = pd.DataFrame()

        for paper in paperList:
            try:
                data = pd.DataFrame(yf.Ticker(paper).dividends)
            except:
                continue
            if not data.empty:
                data["SYMBOL"] = paper.replace(".SA", "")
                res = pd.concat([res, data], axis=0)

        res.reset_index(inplace=True)
        res.rename(columns={"Date": "DATE", "Dividends": "PRICE"}, inplace=True)
        res["PAYDATE"] = res["DATE"] = pd.to_datetime(res["DATE"], format="%Y/%m/%d").dt.tz_localize(None)
        res = res[res["DATE"] >= self.start_date]
        # 30% tax
        res["TAX"] = res["PRICE"] * 0.3 * 0
        # print(res)
        res = res[["SYMBOL", "DATE", "PRICE", "PAYDATE", "TAX"]]
        res["OPERATION"] = "D2"
        return res
