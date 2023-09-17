import re
import os
from glob import glob
import pandas as pd
from collections import namedtuple

from data import DataSchema
from .Broaker import Broaker


class TDAmeritrade(Broaker):
    def __init__(self, out_path):
        self.output = out_path
        super().__init__(os.path.dirname(out_path), os.path.basename(out_path))

    def process_order(self, page):
        text = page.extract_text()

        order = namedtuple("order", "Code Date Company Type Category Qty Value Total Sub Fee")
        line_itens = []
        for line in text.split("\n"):
            res = re.compile(r"YOU\s(BOUGHT|SOLD)\s+(\d+)\s+.+?\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)").search(line)
            if res:
                # print (res.group(0))
                opType = "B" if res.group(1) == "BOUGHT" else "S"
                qty = int(res.group(2))
                value = float(res.group(3))
                fee = float(res.group(5))
                continue

            res = re.compile(r"(\d{2}\/\d{2}\/\d{4})\s+(\d{2}\/\d{2}\/\d{4})\s+([\d.]+)\s+([\d.]+)").search(line)
            if res:
                # print (res.group(0))
                date = pd.to_datetime(res.group(1), format="%m/%d/%Y").strftime("%Y-%m-%d")
                total = res.group(4)
                continue

            res = re.compile(r"^\s(\w+)\s\s\w+(\s\w+)?$").search(line)
            if res:
                # print (res.group(0))
                line_itens.append(order(res.group(1), date, "Company", opType, "Stock", qty, value, total, "sub", fee))
                continue
        self.dtFrame = self.dtFrame.merge(pd.DataFrame(line_itens), how="outer")

    def read_statement(self, in_dir):
        table = pd.DataFrame()
        for file in sorted(glob(in_dir + "/*.csv")):
            df = pd.read_csv(file)
            df = df[~df[DataSchema.DATE].str.contains("END OF FILE")].fillna(0)
            df[DataSchema.TYPE] = "STOCK"
            df[DataSchema.DATE] = pd.to_datetime(df[DataSchema.DATE]).dt.strftime("%Y-%m-%d")
            df = df[
                [
                    DataSchema.SYMBOL,
                    DataSchema.DATE,
                    DataSchema.PRICE,
                    DataSchema.QTY,
                    DataSchema.DESCRIPTION,
                    DataSchema.TYPE,
                    "COMMISSION",
                    DataSchema.AMOUNT,
                ]
            ]

            df.columns = [
                DataSchema.SYMBOL,
                DataSchema.DATE,
                DataSchema.PRICE,
                DataSchema.QTY,
                DataSchema.DESCRIPTION,
                DataSchema.TYPE,
                DataSchema.FEES,
                DataSchema.AMOUNT,
            ]

            df = df.apply(description_parser, axis=1)
            df = df.rename(columns={DataSchema.DESCRIPTION: DataSchema.OPERATION})
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
    if "Bought" in desc:
        row[DataSchema.DESCRIPTION] = "B"
    if "Sold" in desc:
        row[DataSchema.DESCRIPTION] = "S"
    if "DIVIDEND" in desc:
        row[DataSchema.DESCRIPTION] = "D1"
        row[DataSchema.QTY] = 1
        row[DataSchema.PRICE] = row[DataSchema.AMOUNT]
    if "GAIN DISTRIBUTION" in desc:
        row[DataSchema.DESCRIPTION] = "D1"
        row[DataSchema.QTY] = 1
        row[DataSchema.PRICE] = row[DataSchema.AMOUNT]
    if "TAX WITHHELD" in desc:
        row[DataSchema.DESCRIPTION] = "D1"
        row[DataSchema.QTY] = 1
        row[DataSchema.PRICE] = row[DataSchema.AMOUNT]
    if "W-8" in desc:  # Dividend Taxes
        row[DataSchema.DESCRIPTION] = "T1"
        row[DataSchema.QTY] = 1
        row[DataSchema.PRICE] = row[DataSchema.AMOUNT]
    if "REORGANIZATION FEE" in desc:
        row[DataSchema.DESCRIPTION] = "T1"
        row[DataSchema.QTY] = 1
        row[DataSchema.PRICE] = row[DataSchema.AMOUNT]
    if "SPLIT" in desc:
        row[DataSchema.DESCRIPTION] = "SPLIT-TD"
        row[DataSchema.PRICE] = 0
        row[DataSchema.SYMBOL] = ""
    if "WIRE" in desc:
        row[DataSchema.DESCRIPTION] = "C"
        row[DataSchema.TYPE] = "WIRE"
        row[DataSchema.SYMBOL] = DataSchema.CASH
        row[DataSchema.QTY] = 1
        row[DataSchema.PRICE] = row[DataSchema.AMOUNT]
    if "INTEREST" in desc:
        row[DataSchema.DESCRIPTION] = "C"
        row[DataSchema.TYPE] = "INTEREST"
        row[DataSchema.SYMBOL] = DataSchema.CASH
        row[DataSchema.QTY] = 1
        row[DataSchema.PRICE] = row[DataSchema.AMOUNT]
    return row
