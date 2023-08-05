import pandas as pd


class Broaker:
    def __init__(self, outDir, name="default"):
        self.output = outDir + "/" + name + ".csv"
        self.dtFrame = pd.DataFrame(
            columns=["Code", "Date", "Company", "Type", "Category", "Qty", "Value", "Total", "Sub"]
        )

    def process(self, page):
        pass

    def finish(self):
        self.dtFrame.to_csv(self.output, index=False, float_format="%.5f")
