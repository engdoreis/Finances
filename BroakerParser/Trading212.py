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
            df["TYPE"] = "STOCK"
            # df["COMMISSION"] = 0
            df["DATE"] = pd.to_datetime(df["Time"]).dt.strftime("%Y-%m-%d")
            df = df[
                [
                    "Ticker",
                    "DATE",
                    "Price / share",
                    "No. of shares",
                    "Action",
                    "TYPE",
                    "Currency conversion fee",
                    "Total",
                ]
            ]
            df.columns = ["SYMBOL", "DATE", "PRICE", "QUANTITY", "Action", "TYPE", "COMMISSION", "AMOUNT"]

            df.fillna(0, inplace=True)
            df["PRICE"] = df["AMOUNT"] / df["QUANTITY"]
            df = df.apply(action_process, axis=1)
            df = df.rename(columns={"Action": "OPERATION"})
            if table.empty:
                table = pd.concat([table, df])
            else:
                table = table.merge(
                    df,
                    how="outer",
                    on=["SYMBOL", "DATE", "PRICE", "QUANTITY", "OPERATION", "TYPE", "COMMISSION", "AMOUNT"],
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
        row["QUANTITY"] = 1
        row["PRICE"] = row["AMOUNT"]
    elif "deposit" in action:
        row["Action"] = "C"
        row["TYPE"] = "WIRE"
        row["SYMBOL"] = "CASH"
        row["QUANTITY"] = 1
        row["PRICE"] = row["AMOUNT"]
    elif "interest" in action:
        row["Action"] = "C"
        row["TYPE"] = "INTEREST"
        row["SYMBOL"] = "CASH"
        row["QUANTITY"] = 1
        row["PRICE"] = row["AMOUNT"]
    else:
        print(f'Action "{action}" is unknown.')
    return row
