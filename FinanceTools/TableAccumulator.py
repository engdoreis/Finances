import numpy as np
import pandas as pd


class TableAccumulator:
    cash = avr = brl_avr = qty_cum = prov_cum = 0

    def __init__(self, pcr=None):
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
            self.avr = (self.avr * self.qty_cum) + operationValue
            self.brl_avr = (self.brl_avr * self.qty_cum) + (operationValue / currency_rate)
            self.qty_cum += qty
            self.avr /= self.qty_cum
            self.brl_avr /= self.qty_cum

        # Sell
        elif stType == "S":
            self.qty_cum += qty
            if self.qty_cum == 0:
                self.prov_cum = 0

        # Amortization
        elif stType in ["A"]:
            total = np.nan
            row["QUANTITY"] = self.qty_cum
            if self.qty_cum > 0:
                operationValue = row.loc["PRICE"] * self.qty_cum + row.loc["FEE"]
                self.avr = ((self.avr * self.qty_cum) - operationValue) / self.qty_cum
                total = row.loc["PRICE"] * self.qty_cum
                self.prov_cum += total

        # Split
        elif stType == "SPLIT":
            self.qty_cum *= qty
            self.avr /= qty

        # Dividend
        elif stType in ["D", "R", "JCP"]:
            total = np.nan
            if row["QUANTITY"] == 0 and self.qty_cum != 0:
                # Means the price represents the total
                row["PRICE"] /= self.qty_cum

            row["QUANTITY"] = self.qty_cum
            if self.qty_cum > 0:
                total = row.loc["PRICE"] * row["QUANTITY"]
                self.prov_cum += total

        # Dividend, Tax, Amortization
        elif stType in ["D1", "R1", "JCP1", "T1", "A1", "I1"]:
            total = row.loc["PRICE"] * row["QUANTITY"]
            if stType != "I1":
                self.prov_cum += total

        row["AMOUNT"] = total
        row["acumProv"] = self.prov_cum
        row["acum_qty"] = self.qty_cum
        row["PM"] = self.avr
        row["PM_BRL"] = self.brl_avr
        return row

    def ByGroup(self, group):
        self.avr = self.brl_avr = self.qty_cum = self.prov_cum = 0
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
