import os
from glob import glob
import pandas as pd

from .Broaker import Broaker
from FinanceTools import *


class Trading212(Broaker):
    def __init__(self, out_path):
        self.output = out_path
        super().__init__(os.path.dirname(out_path), os.path.basename(out_path))

    def process_order(self, page):
        pass

    def read_statement(self, in_dir):
        table = pd.DataFrame()
        for file in sorted(glob(in_dir + "/*.csv")):
            df = pd.read_csv(file)

            df = df.apply(isin_process, axis=1)
            df[DataSchema.TYPE] = "STOCK"
            # df["COMMISSION"] = 0
            df[DataSchema.DATE] = pd.to_datetime(df["Time"]).dt.strftime("%Y-%m-%d")
            df = df[
                [
                    "Ticker",
                    DataSchema.DATE,
                    "Price / share",
                    "No. of shares",
                    "Action",
                    DataSchema.TYPE,
                    "Currency conversion fee",
                    "Total",
                ]
            ]
            df.columns = [
                DataSchema.SYMBOL,
                DataSchema.DATE,
                DataSchema.PRICE,
                DataSchema.QTY,
                "Action",
                DataSchema.TYPE,
                DataSchema.FEES,
                DataSchema.AMOUNT,
            ]

            df.fillna(0, inplace=True)
            df[DataSchema.PRICE] = df[DataSchema.AMOUNT] / df[DataSchema.QTY]
            df = df.apply(action_process, axis=1)
            df = df.rename(columns={"Action": DataSchema.OPERATION})
            if table.empty:
                table = pd.concat([table, df])
            else:
                table = table.merge(
                    df,
                    how="outer",
                    on=[
                        DataSchema.SYMBOL,
                        DataSchema.DATE,
                        DataSchema.PRICE,
                        DataSchema.QTY,
                        DataSchema.OPERATION,
                        DataSchema.TYPE,
                        DataSchema.FEES,
                        DataSchema.AMOUNT,
                    ],
                    suffixes=["", "_"],
                    indicator=True,
                )
                table.drop(["_merge"], axis=1, inplace=True)
                table = table.loc[:, ~table.columns.str.endswith("_")]
        table.to_csv(self.output, index=False)


def isin_process(row):
    isin = str(row["ISIN"])
    currency = row["Currency (Price / share)"]
    if isin.startswith("GB") or currency == "GBP":
        row["Ticker"] += ".L"

    # FIX: This is handling an exception when a free share is earned. This can make US shares to have UK ISIN
    sdrt = row.get("Stamp duty reserve tax", 0)
    if sdrt > 0:
        row["Ticker"] = row["Ticker"].replace(".L", "")
    return row


def action_process(row):
    action = row["Action"].lower()
    if "buy" in action:
        row["Action"] = "B"
    elif "sell" in action:
        row["Action"] = "S"
    elif "dividend" in action:
        row["Action"] = "D1"
        row[DataSchema.QTY] = 1
        row[DataSchema.PRICE] = row[DataSchema.AMOUNT]
    elif "deposit" in action:
        row["Action"] = "C"
        row[DataSchema.TYPE] = "WIRE"
        row[DataSchema.SYMBOL] = DataSchema.CASH
        row[DataSchema.QTY] = 1
        row[DataSchema.PRICE] = row[DataSchema.AMOUNT]
    elif "interest" in action:
        row["Action"] = "C"
        row[DataSchema.TYPE] = "INTEREST"
        row[DataSchema.SYMBOL] = DataSchema.CASH
        row[DataSchema.QTY] = 1
        row[DataSchema.PRICE] = row[DataSchema.AMOUNT]
    else:
        print(f'Action "{action}" is unknown.')
    return row
