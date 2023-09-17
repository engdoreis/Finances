from dataclasses import dataclass

from data import DataSchema

""" Class to calculate the average price by Stock group"""


@dataclass
class Acumulator:
    acumulated: int = 0

    def calcLoss(self, row):
        acumulated = self.acumulated

        if row.loc[DataSchema.PROFIT] < 0 or acumulated < 0:
            acumulated = acumulated + row.loc[DataSchema.PROFIT]

        if acumulated > 0:
            acumulated = 0

        self.acumulated = acumulated
        return self.acumulated
