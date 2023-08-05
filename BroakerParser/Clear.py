import re

import pandas as pd

from .Broaker import Broaker


class Clear(Broaker):
    def __init__(self, outDir, name="default"):
        super().__init__(outDir, name)

    def process(self, page):
        liqFee = liqFee = emolFee = opFee = exFee = custodyFee = irrf = taxes = otherFee = 0

        text = page.extract_text()
        order = namedtuple("order", "Code Date Company Type Category Qty Value Total Sub ")
        line_itens = []
        for line in text.split("\n"):
            res = re.compile(
                r"[\w\d-]+\s(C|V)\s+(?:VISTA|FRACIONARIO)\s(?:\d\d\/\d\d)?([\w\d\s.\/]+?)\s\s+([\w\d\s#]+?)\s([\d.,]+)\s([\d.,]+)\s([\d.,]+)\s(\w)"
            ).search(line)
            if res:
                # print (res.group(0))
                opType = "S" if res.group(1) == "V" else "B"
                name = res.group(2).strip()
                code = res.group(3).strip()

                if ("FII" in name) or ("FDO" in name):
                    category = "FII"
                    code = code.split(" ")[0]
                else:
                    category = "Ação"

                line_itens.append(
                    order(
                        code,
                        Date,
                        name,
                        opType,
                        category,
                        int(res.group(4).replace(".", "")),
                        to_float(res.group(5)),
                        to_float(res.group(6)),
                        code.split(" ")[0],
                    )
                )
                continue

            res = re.compile(r"\d{2}/\d{2}/\d{4}$").search(line)
            if res:
                Date = pd.to_datetime(res.group(0), format="%d/%m/%Y").strftime("%Y-%m-%d")
                continue

            res = re.compile(r".*?Taxa de liquida.*?\s+([\d,]+)").search(line)
            if res:
                liqFee = to_float(res.group(1))
                continue

            res = re.compile(r"Emolumentos\s+([\d,]+)").search(line)
            if res:
                emolFee = to_float(res.group(1))
                continue

            res = re.compile(r"Taxa Operacional\s+([\d,]+)").search(line)
            if res:
                opFee = to_float(res.group(1))
                continue

            res = re.compile(r"Execu\w+\s+([\d,]+)").search(line)
            if res:
                exFee = to_float(res.group(1))
                continue

            res = re.compile(r".*?Taxa de Cust\w+\s+([\d,]+)").search(line)
            if res:
                custodyFee = to_float(res.group(1))
                continue

            res = re.compile(r"I.R.R.F.*?base.*?[\d,]+\s([\d,]+)").search(line)
            if res:
                irrf = to_float(res.group(1))
                continue

            res = re.compile(r"Impostos\s+([\d,]+)").search(line)
            if res:
                taxes = to_float(res.group(1))
                continue

            res = re.compile(r"Outros\s+([\d,]+)").search(line)
            if res:
                otherFee = to_float(res.group(1))
                continue

        df = pd.DataFrame(line_itens)

        total = df["Total"].sum()
        df["LiqFee"] = liqFee * df["Total"] / total
        df["EmolFee"] = emolFee * df["Total"] / total
        df["OpFee"] = opFee * df["Total"] / total
        df["ExFee"] = exFee * df["Total"] / total
        df["CustodyFee"] = custodyFee * df["Total"] / total
        df["Irrf"] = irrf * df["Total"] / total
        df["Taxes"] = taxes * df["Total"] / total
        df["otherFee"] = otherFee * df["Total"] / total
        df["Fee"] = (
            df["LiqFee"] + df["EmolFee"] + df["OpFee"] + df["ExFee"] + df["CustodyFee"] + df["Taxes"] + df["otherFee"]
        )
        self.dtFrame = self.dtFrame.merge(df, how="outer")
