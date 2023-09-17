import numpy as np
import pandas as pd

from data import DataSchema


class TableAccumulator:
    cash = avr = brl_avr = qty_cum = prov_cum = 0

    def __init__(self, pcr=None, currency="USD"):
        self.pcr = pcr
        self.currency = currency

    def get_currency_rate(self, date: str):
        return (
            1
            if self.currency == "USD"
            else self.pcr.getIndexCurrentValue(self.currency + "USD", date)
            if self.pcr
            else 1
        )

    def ByRow(self, row):
        total = row.loc[DataSchema.AMOUNT]
        stType = row.loc[DataSchema.OPERATION]
        qty = row.loc[DataSchema.QTY]
        currency_rate = self.get_currency_rate(row[DataSchema.DATE])

        # buy
        if stType == "B":
            operationValue = row.loc[DataSchema.PRICE] * qty + row.loc[DataSchema.FEES]
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
            row[DataSchema.QTY] = self.qty_cum
            if self.qty_cum > 0:
                operationValue = row.loc[DataSchema.PRICE] * self.qty_cum + row.loc[DataSchema.FEES]
                self.avr = ((self.avr * self.qty_cum) - operationValue) / self.qty_cum
                total = row.loc[DataSchema.PRICE] * self.qty_cum
                self.prov_cum += total

        # Split
        elif stType == "SPLIT":
            self.qty_cum *= qty
            self.avr /= qty

        # Dividend
        elif stType in ["D", "R", "JCP"]:
            total = np.nan
            if row[DataSchema.QTY] == 0 and self.qty_cum != 0:
                # Means the price represents the total
                row[DataSchema.PRICE] /= self.qty_cum

            row[DataSchema.QTY] = self.qty_cum
            if self.qty_cum > 0:
                total = row.loc[DataSchema.PRICE] * row[DataSchema.QTY]
                self.prov_cum += total

        # Dividend, Tax, Amortization
        elif stType in ["D1", "R1", "JCP1", "T1", "A1", "I1"]:
            total = row.loc[DataSchema.PRICE] * row[DataSchema.QTY]
            if stType != "I1":
                self.prov_cum += total

        row[DataSchema.AMOUNT] = total
        row[DataSchema.DIV_ACUM] = self.prov_cum
        row[DataSchema.QTY_ACUM] = self.qty_cum
        row[DataSchema.AVERAGE_PRICE] = self.avr
        row[DataSchema.PM_BRL] = self.brl_avr
        return row

    def ByGroup(self, group):
        self.avr = self.brl_avr = self.qty_cum = self.prov_cum = 0
        return group.apply(self.ByRow, axis=1)

    def Cash(self, row):
        stType = row.loc[DataSchema.OPERATION]
        amount = round(row.loc[DataSchema.AMOUNT], 6)

        if stType in ["C", "W"]:
            self.cash += amount + row.loc[DataSchema.FEES]
            row.loc[DataSchema.QTY_ACUM] = row.loc[DataSchema.QTY]
            row.loc[DataSchema.AVERAGE_PRICE] = row.loc[DataSchema.PRICE]
            row[DataSchema.PM_BRL] = row.loc[DataSchema.PRICE] / self.get_currency_rate(row[DataSchema.DATE])

        elif stType in ["B", "S"]:
            self.cash -= amount + row.loc[DataSchema.FEES]

        elif (stType in ["D1", "A1", "R1", "JCP1", "T1", "I1", "CF"]) or (
            stType in ["D", "A", "R", "JCP", "T"] and row[DataSchema.QTY_ACUM] > 0
        ):
            self.cash += amount

        row[DataSchema.CASH] = round(self.cash, 6)
        return row
