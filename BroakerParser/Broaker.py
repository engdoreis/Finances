import pandas as pd


class Broaker:
    def __init__(self, outDir, name="default.csv"):
        self.output = outDir + "/" + name
        self.dtFrame = pd.DataFrame(
            columns=["Code", "Date", "Company", "Type", "Category", "Qty", "Value", "Total", "Sub"]
        )

    def process_order(self, page):
        pass

    def finish(self):
        self.dtFrame.to_csv(self.output, index=False, float_format="%.5f")
