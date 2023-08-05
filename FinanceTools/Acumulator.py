# Class to calculate the average price by Stock group
class Acumulator:
    def __init__(self):
        self.acumulated = 0

    def calcLoss(self, row):
        acumulated = self.acumulated

        if row.loc["Profit"] < 0 or acumulated < 0:
            acumulated = acumulated + row.loc["Profit"]

        if acumulated > 0:
            acumulated = 0

        self.acumulated = acumulated
        return self.acumulated
