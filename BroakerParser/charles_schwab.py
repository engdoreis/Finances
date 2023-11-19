import re
import os
from glob import glob
import pandas as pd
from collections import namedtuple

from data import DataSchema
from .Broaker import Broaker


class CharlesChwab(Broaker):
    def __init__(self, out_path):
        self.output = out_path
        super().__init__(os.path.dirname(out_path), os.path.basename(out_path))

    def process_order(self, page):
        pass

    def read_statement(self, in_dir):
        table = pd.DataFrame()
        for file in sorted(glob(in_dir + "/*.csv")):
            df = pd.read_csv(file)
            df.columns = [
                DataSchema.DATE,
                DataSchema.OPERATION,
                DataSchema.SYMBOL,
                DataSchema.DESCRIPTION,
                DataSchema.QTY,
                DataSchema.PRICE,
                DataSchema.FEES,
                DataSchema.AMOUNT,
            ]

            df = df[~df[DataSchema.DATE].str.contains("Transactions Total")].fillna(0)
            df[DataSchema.DATE] = pd.to_datetime(df[DataSchema.DATE]).dt.strftime("%Y-%m-%d")
            df[DataSchema.TYPE] = "STOCK"

            df = df[
                [
                    DataSchema.SYMBOL,
                    DataSchema.DATE,
                    DataSchema.OPERATION,
                    DataSchema.PRICE,
                    DataSchema.QTY,
                    DataSchema.DESCRIPTION,
                    DataSchema.TYPE,
                    DataSchema.FEES,
                    DataSchema.AMOUNT,
                ]
            ]

            df = df.apply(description_parser, axis=1)
            df = df[df[DataSchema.SYMBOL] != "INTERNAL"]
            df = df.replace("[\$]", "", regex=True)
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


def description_parser(row):
    desc = row[DataSchema.DESCRIPTION]
    operation = row[DataSchema.OPERATION]
    if "Buy" in operation:
        row[DataSchema.OPERATION] = "B"
        if row[DataSchema.SYMBOL].isnumeric():
            row[DataSchema.SYMBOL] = desc.split("(")[1].split(")")[0]
    elif "Sell" in operation:
        row[DataSchema.OPERATION] = "S"
    elif "Dividend" in operation or "DIVIDEND" in desc or "GAIN DISTRIBUTION" in desc:
        row[DataSchema.OPERATION] = "D1"
        row[DataSchema.QTY] = 1
        row[DataSchema.PRICE] = row[DataSchema.AMOUNT]
        if "DIVIDEND" in desc or "GAIN DISTRIBUTION" in desc:
            row[DataSchema.SYMBOL] = desc.split("(")[1].split(")")[0]
    elif "Interest" in operation:
        row[DataSchema.OPERATION] = "C"
        row[DataSchema.TYPE] = "INTEREST"
        row[DataSchema.SYMBOL] = DataSchema.CASH
        row[DataSchema.QTY] = 1
        row[DataSchema.PRICE] = row[DataSchema.AMOUNT]
    elif "Tax" in operation or "W-8" in desc:  # Dividend Taxes
        row[DataSchema.OPERATION] = "T1"
        row[DataSchema.QTY] = 1
        row[DataSchema.PRICE] = row[DataSchema.AMOUNT]
        if "INT" in desc:
            row[DataSchema.SYMBOL] = DataSchema.CASH
        if "TDA TRAN" in desc:
            row[DataSchema.SYMBOL] = desc.split("(")[1].split(")")[0]
    elif "Wire" in operation:
        row[DataSchema.OPERATION] = "C"
        row[DataSchema.TYPE] = "WIRE"
        row[DataSchema.SYMBOL] = DataSchema.CASH
        row[DataSchema.QTY] = 1
        row[DataSchema.PRICE] = row[DataSchema.AMOUNT]
    elif "REORGANIZATION FEE" in desc:
        row[DataSchema.OPERATION] = "T1"
        row[DataSchema.QTY] = 1
        row[DataSchema.PRICE] = row[DataSchema.AMOUNT]
        row[DataSchema.SYMBOL] = DataSchema.CASH
    elif "SPLIT" in desc:
        row[DataSchema.OPERATION] = "SPLIT-TD"
        row[DataSchema.PRICE] = 0
    elif operation in ["Journaled Shares", "Internal Transfer"]:
        row[DataSchema.OPERATION] = "INTERNAL"
        row[DataSchema.TYPE] = "INTERNAL"
        row[DataSchema.SYMBOL] = "INTERNAL"
    else:
        print(row)
        exit(1)
    return row
