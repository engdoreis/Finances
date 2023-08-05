import re

import pandas as pd

from .Broaker import Broaker


class TDAmeritrade(Broaker):
    def __init__(self, outDir, name="default"):
        self.output = outDir + "/" + name + ".csv"
        super().__init__(outDir, name)

    def process(self, page):
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
