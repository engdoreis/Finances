import re
from collections import namedtuple


class ClearProventos:
    def __init__(self, input, outDir, name="default"):
        self.input = input
        self.output = outDir + "/" + name + ".csv"
        self.dtFrame = pd.DataFrame(
            columns=list("Ativo Evento Quantidade ValorBruto ValorIR ValorLiquido Pagamento".split())
        )
        self.dividend_re = re.compile(r"DIVIDENDOS\s([\d]+)\s")

    def process(self, page):
        text = page.extract_text()

        order = namedtuple("order", "Ativo Evento Quantidade ValorBruto ValorIR ValorLiquido Pagamento")
        line_itens = []
        for line in text.split("\n"):
            res = re.compile(
                r"([a-zA-Z/. ]+)\s([a-zA-Z/]+)\s([0-9,.]+)\s([0-9,.]+)\s([0-9,.]+)\s([0-9,.]+)\s([0-9/]+)"
            ).search(line)
            if res:
                # print (res.group(0))
                line_itens.append(
                    order(
                        res.group(1), res.group(2), res.group(3), res.group(4), res.group(5), res.group(6), res.group(7)
                    )
                )
                continue
            # print(line)

        if len(line_itens) > 0:
            self.dtFrame = self.dtFrame.merge(pd.DataFrame(line_itens), how="outer")

    def finish(self):
        self.dtFrame.to_csv(self.output, index=False, float_format="%.5f")
