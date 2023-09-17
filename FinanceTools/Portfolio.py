import numpy as np
import pandas as pd

from .Color import Color
from .TableAccumulator import TableAccumulator


from data import DataSchema


class Portfolio:
    def __init__(self, price_reader, split_reader, date, dFrame, recommended=None, currency="$"):
        self.currency = currency
        self.dtframe = dFrame.groupby([DataSchema.SYMBOL]).apply(lambda x: x.tail(1))

        dFrame = dFrame.sort_values([DataSchema.PAYDATE, DataSchema.OPERATION], ascending=[True, False])
        dFrame = dFrame.apply(TableAccumulator().Cash, axis=1)
        cash = dFrame.iloc[-1][DataSchema.CASH]

        self.dtframe = self.dtframe[
            [DataSchema.SYMBOL, DataSchema.AVERAGE_PRICE, DataSchema.QTY_ACUM, DataSchema.DIV_ACUM, DataSchema.TYPE]
        ]
        self.dtframe.columns = [
            DataSchema.SYMBOL,
            DataSchema.AVERAGE_PRICE,
            DataSchema.QTY,
            "DIVIDENDS",
            DataSchema.TYPE,
        ]
        self.dtframe["COST"] = self.dtframe.PM * self.dtframe[DataSchema.QTY]
        self.dtframe = self.dtframe[self.dtframe[DataSchema.QTY] > 0]
        self.dtframe.reset_index(drop=True, inplace=True)

        self.dtframe = self.dtframe[self.dtframe[DataSchema.SYMBOL] != DataSchema.CASH]

        def fillCurrentValue(pr, sr, date, row):
            return pr.getCurrentValue(row[DataSchema.SYMBOL], date) * sr.get_accumulated(row[DataSchema.SYMBOL], date)

        self.dtframe[DataSchema.PRICE] = self.dtframe.apply(
            lambda row: fillCurrentValue(price_reader, split_reader, date, row), axis=1
        )

        self.dtframe[DataSchema.PRICE] = self.dtframe[DataSchema.PRICE].fillna(self.dtframe[DataSchema.AVERAGE_PRICE])
        self.dtframe["MKT_VALUE"] = self.dtframe[DataSchema.PRICE] * self.dtframe[DataSchema.QTY]

        newLine = {
            DataSchema.SYMBOL: DataSchema.CASH,
            DataSchema.AVERAGE_PRICE: cash,
            DataSchema.QTY: 1,
            "DIVIDENDS": 0,
            DataSchema.TYPE: "C",
            "COST": cash,
            DataSchema.PRICE: cash,
            "MKT_VALUE": cash,
        }
        self.dtframe = pd.concat([self.dtframe, pd.DataFrame(newLine, index=[0])])

        self.dtframe[f"GAIN({currency})"] = self.dtframe["MKT_VALUE"] - self.dtframe["COST"]
        self.dtframe[f"GAIN+DIV({currency})"] = self.dtframe[f"GAIN({currency})"] + self.dtframe["DIVIDENDS"]
        self.dtframe["GAIN(%)"] = self.dtframe[f"GAIN({currency})"] / self.dtframe["COST"]
        self.dtframe["GAIN+DIV(%)"] = self.dtframe[f"GAIN+DIV({currency})"] / self.dtframe["COST"]
        self.dtframe["ALLOCATION"] = self.dtframe["MKT_VALUE"] / self.dtframe["MKT_VALUE"].sum()
        self.dtframe = self.dtframe.replace([np.inf, -np.inf], np.nan).fillna(0)

        self.dtframe = self.dtframe[self.dtframe[DataSchema.AVERAGE_PRICE] > 0]

        self.dtframe = self.dtframe[
            [
                DataSchema.SYMBOL,
                DataSchema.AVERAGE_PRICE,
                DataSchema.PRICE,
                DataSchema.QTY,
                "COST",
                "MKT_VALUE",
                "DIVIDENDS",
                f"GAIN({currency})",
                f"GAIN+DIV({currency})",
                "GAIN(%)",
                "GAIN+DIV(%)",
                "ALLOCATION",
            ]
        ]

        self.format = {
            DataSchema.PRICE: f"{currency} {{:,.2f}}",
            DataSchema.AVERAGE_PRICE: f"{currency} {{:,.2f}}",
            DataSchema.QTY: "{:>n}",
            "COST": f"{currency} {{:,.2f}}",
            "MKT_VALUE": f"{currency} {{:,.2f}}",
            "DIVIDENDS": f"{currency} {{:,.2f}}",
            f"GAIN({currency})": f"{currency} {{:,.2f}}",
            f"GAIN+DIV({currency})": f"{currency} {{:,.2f}}",
            "GAIN(%)": "{:,.2f}%",
            "GAIN+DIV(%)": "{:,.2f}%",
            "ALLOCATION": "{:,.2f}%",
        }

        self.extra_content(recommended)

        self.dtframe.set_index(DataSchema.SYMBOL, inplace=True)

    def extra_content(self, recommended):
        if recommended == None:
            return

        self.dtframe["TARGET"], self.dtframe["TOP_PRICE"], self.dtframe["PRIORITY"] = zip(
            *self.dtframe[DataSchema.SYMBOL].map(lambda x: self.recommended(recommended, x))
        )
        self.dtframe["BUY"] = (
            self.dtframe[DataSchema.QTY] * (self.dtframe["TARGET"] - self.dtframe["ALLOCATION"])
        ) / self.dtframe["ALLOCATION"]

        format = {"TARGET": "{:,.2f}%", "TOP_PRICE": f"{self.currency} {{:,.2f}}", "BUY": "{:,.1f}"}
        self.format = {**self.format, **format}

    def recommended(self, recom, symbol):
        for ticker in recom["Tickers"]:
            if symbol == ticker["Ticker"]:
                return float(ticker["Participation"]), float(ticker["Top"]), int(ticker["Priority"])
        return 0, 0, 99

    def get_table(self):
        return self.dtframe

    def show(self):
        fdf = self.dtframe.copy(deep=True)
        return fdf.style.applymap(Color().color_negative_red).format(self.format)
