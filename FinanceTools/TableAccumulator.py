import pandas as pd
import numpy as np


class TableAccumulator:
    def __init__(self, pcr=None):
        self.cash = self.avr = self.brl_avr = self.acumQty = self.acumProv = 0
        self.pcr = pcr

    def get_currency_rate(self, date):
        currency_rate = 1
        if not self.pcr == None:
            currency_rate = self.pcr.getIndexCurrentValue("USD", date)
        return currency_rate

    def ByRow(self, row):
        total = row.loc["AMOUNT"]
        stType = row.loc["OPERATION"]
        qty = row.loc["QUANTITY"]
        currency_rate = self.get_currency_rate(row["DATE"])

        # buy
        if stType == "B":
            operationValue = row.loc["PRICE"] * qty + row.loc["FEE"]
            self.avr = (self.avr * self.acumQty) + operationValue
            self.brl_avr = (self.brl_avr * self.acumQty) + (operationValue / currency_rate)
            self.acumQty += qty
            self.avr /= self.acumQty
            self.brl_avr /= self.acumQty

        # Sell
        elif stType == "S":
            self.acumQty += qty
            if self.acumQty == 0:
                self.acumProv = 0

        # Amortization
        elif stType in ["A"]:
            total = np.nan
            row["QUANTITY"] = self.acumQty
            if self.acumQty > 0:
                operationValue = row.loc["PRICE"] * self.acumQty + row.loc["FEE"]
                self.avr = ((self.avr * self.acumQty) - operationValue) / self.acumQty
                total = row.loc["PRICE"] * self.acumQty
                self.acumProv += total

        # Split
        elif stType == "SPLIT":
            self.acumQty *= qty
            self.avr /= qty

        # Dividend
        elif stType in ["D", "R", "JCP"]:
            total = np.nan
            if row["QUANTITY"] == 0 and self.acumQty != 0:
                # Means the price represents the total
                row["PRICE"] /= self.acumQty

            row["QUANTITY"] = self.acumQty
            if self.acumQty > 0:
                total = row.loc["PRICE"] * row["QUANTITY"]
                self.acumProv += total

        # Dividend, Tax, Amortization
        elif stType in ["D1", "R1", "JCP1", "T1", "A1", "I1"]:
            total = row.loc["PRICE"] * row["QUANTITY"]
            if stType != "I1":
                self.acumProv += total

        row["AMOUNT"] = total
        row["acumProv"] = self.acumProv
        row["acum_qty"] = self.acumQty
        row["PM"] = self.avr
        row["PM_BRL"] = self.brl_avr
        return row

    def ByGroup(self, group):
        self.avr = self.brl_avr = self.acumQty = self.acumProv = 0
        return group.apply(self.ByRow, axis=1)

    def Cash(self, row):
        stType = row.loc["OPERATION"]
        amount = round(row.loc["AMOUNT"], 6)

        if stType in ["C", "W"]:
            self.cash += amount + row.loc["FEE"]
            row.loc["acum_qty"] = row.loc["QUANTITY"]
            row.loc["PM"] = row.loc["PRICE"]
            row["PM_BRL"] = row.loc["PRICE"] / self.get_currency_rate(row["DATE"])

        elif stType in ["B", "S"]:
            self.cash -= amount + row.loc["FEE"]

        elif (stType in ["D1", "A1", "R1", "JCP1", "T1", "I1", "CF"]) or (
            stType in ["D", "A", "R", "JCP", "T"] and row["acum_qty"] > 0
        ):
            # self.acumProv += amount
            self.cash += amount

        row["CASH"] = round(self.cash, 6)
        return row
