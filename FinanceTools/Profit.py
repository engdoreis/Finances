# Class to calculate the profit or loss considering day trade rules.
import pandas as pd


class Profit:
    def __init__(self):
        self.pm = self.amount = 0

    def DayTrade(self, row):
        profit = 0
        amount = self.amount + row.QUANTITY
        if row.OPERATION == "B":
            self.pm = (row.PRICE * row.QUANTITY) / amount
        else:
            profit = (self.pm - row.PRICE) * row.QUANTITY
            amount = self.amount - row.QUANTITY

        self.amount = amount
        row["Profit"] = profit
        row["DayTrade"] = 1
        return row

    def Trade(self, dayGroup):
        purchaseDf = dayGroup.loc[dayGroup.OPERATION == "B"]
        sellDf = dayGroup.loc[dayGroup.OPERATION == "S"]

        sellCount = len(sellDf)
        purchaseCount = len(purchaseDf)

        if sellCount == 0:
            dayGroup["Profit"] = dayGroup["DayTrade"] = 0
            return dayGroup

        if purchaseCount == 0:
            dayGroup["Profit"] = ((dayGroup.PRICE - dayGroup.PM) * -dayGroup.QUANTITY) - dayGroup.FEE
            dayGroup["DayTrade"] = 0
            return dayGroup

        # Day trade detected
        # print('Day Trade detected\n', dayGroup)
        self.pm = self.amount = 0
        return dayGroup.apply(self.DayTrade, axis=1)
