import pandas as pd
import numpy as np

from .Color import Color
from .TableAccumulator import TableAccumulator


class Portifolio:
    def __init__(self, priceReader, dFrame, recommended=None, currency="$"):
        self.currency = currency
        self.dtframe = dFrame.groupby(["SYMBOL"]).apply(lambda x: x.tail(1))

        dFrame = dFrame.sort_values(["PAYDATE", "OPERATION"], ascending=[True, False])
        dFrame = dFrame.apply(TableAccumulator().Cash, axis=1)
        cash = dFrame.iloc[-1]["CASH"]

        self.dtframe = self.dtframe[["SYMBOL", "PM", "acum_qty", "acumProv", "TYPE"]]
        self.dtframe.columns = ["SYMBOL", "PM", "QUANTITY", "DIVIDENDS", "TYPE"]
        self.dtframe["COST"] = self.dtframe.PM * self.dtframe["QUANTITY"]
        self.dtframe = self.dtframe[self.dtframe["QUANTITY"] > 0]
        self.dtframe.reset_index(drop=True, inplace=True)

        self.dtframe = self.dtframe[self.dtframe["SYMBOL"] != "CASH"]
        self.dtframe["PRICE"] = self.dtframe.apply(priceReader.fillCurrentValue, axis=1)["PRICE"]
        self.dtframe["PRICE"] = self.dtframe["PRICE"].fillna(self.dtframe["PM"])
        self.dtframe["MKT_VALUE"] = self.dtframe["PRICE"] * self.dtframe["QUANTITY"]

        newLine = {
            "SYMBOL": "CASH",
            "PM": cash,
            "QUANTITY": 1,
            "DIVIDENDS": 0,
            "TYPE": "C",
            "COST": cash,
            "PRICE": cash,
            "MKT_VALUE": cash,
        }
        self.dtframe = pd.concat([self.dtframe, pd.DataFrame(newLine, index=[0])])

        self.dtframe[f"GAIN({currency})"] = self.dtframe["MKT_VALUE"] - self.dtframe["COST"]
        self.dtframe[f"GAIN+DIV({currency})"] = self.dtframe[f"GAIN({currency})"] + self.dtframe["DIVIDENDS"]
        self.dtframe["GAIN(%)"] = self.dtframe[f"GAIN({currency})"] / self.dtframe["COST"] * 100
        self.dtframe["GAIN+DIV(%)"] = self.dtframe[f"GAIN+DIV({currency})"] / self.dtframe["COST"] * 100
        self.dtframe["ALLOCATION"] = (self.dtframe["MKT_VALUE"] / self.dtframe["MKT_VALUE"].sum()) * 100
        self.dtframe = self.dtframe.replace([np.inf, -np.inf], np.nan).fillna(0)

        self.dtframe = self.dtframe[self.dtframe["PM"] > 0]

        self.dtframe = self.dtframe[
            [
                "SYMBOL",
                "PM",
                "PRICE",
                "QUANTITY",
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
            "PRICE": f"{currency} {{:,.2f}}",
            "PM": f"{currency} {{:,.2f}}",
            "QUANTITY": "{:>n}",
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

        self.dtframe.set_index("SYMBOL", inplace=True)

    def extra_content(self, recommended):
        if recommended == None:
            return

        self.dtframe["TARGET"], self.dtframe["TOP_PRICE"], self.dtframe["PRIORITY"] = zip(
            *self.dtframe["SYMBOL"].map(lambda x: self.recommended(recommended, x))
        )
        self.dtframe["BUY"] = (
            self.dtframe["QUANTITY"] * (self.dtframe["TARGET"] / 100 - self.dtframe["ALLOCATION"] / 100)
        ) / (self.dtframe["ALLOCATION"] / 100)
        format = {"TARGET": "{:,.2f}%", "TOP_PRICE": f"{self.currency} {{:,.2f}}", "BUY": "{:,.1f}"}
        self.format = {**self.format, **format}

    def recommended(self, recom, symbol):
        for ticker in recom["Tickers"]:
            if symbol == ticker["Ticker"]:
                return float(ticker["Participation"]) * 100, float(ticker["Top"]), int(ticker["Priority"])
        return 0, 0, 99

    def show(self):
        fdf = self.dtframe
        return fdf.style.applymap(Color().color_negative_red).format(self.format)
