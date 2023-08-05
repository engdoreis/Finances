import json
from datetime import datetime
import pandas as pd
from collections import namedtuple
from FinanceTools import *
import numpy as np
import sys


class StatusInvest:
    def __init__(self, inFile):
        self.orders = pd.read_csv(inFile)

        self.orders["Category"] = self.orders["Category"].apply(lambda x: "Ações" if x == "Stock" else "FII's")
        self.orders["Type"] = self.orders["Type"].apply(lambda x: "C" if x == "Compra" else "V")
        self.orders["Date"] = pd.to_datetime(self.orders["Date"]).dt.strftime("%d/%m/%Y")
        self.orders["Corretora"] = "Clear"
        self.orders["Corretagem"] = 0
        self.orders["Impostos"] = 0
        self.orders["IRRF"] = 0

        self.orders = self.orders[
            ["Date", "Category", "Paper", "Type", "Qty", "Value", "Corretora", "Corretagem", "Fee", "Impostos", "IRRF"]
        ]
        self.orders.columns = [
            "Data operação",
            "Categoria",
            "Código Ativo",
            "Operação C/V",
            "Quantidade",
            "Preço unitário",
            "Corretora",
            "Corretagem",
            "Taxas",
            "Impostos",
            "IRRF",
        ]

    def Dataframe(self):
        return self.orders


def Convert(inFile="orders.csv", outFile="orders.csv"):
    si = StatusInvest(inFile)
    si.Dataframe().to_excel(outFile, index=False)


if __name__ == "__main__":
    inFile = str(sys.argv[1])
    outFile = str(sys.argv[2])
    Convert(inFile, outFile)
