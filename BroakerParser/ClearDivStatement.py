import re
from glob import glob

import pandas as pd
from data import DataSchema


class ClearDivStatement:
    def __init__(self, inputDir, outDir, name="default"):
        self.inputDir = inputDir
        self.output = outDir + "/" + name + ".tsv"
        self.dtFrame = pd.DataFrame()
        self.operation_map = {
            "PAGAMENTO DE AMORTIZACAO": "A1",
            "JUROS S/CAPITAL": "JCP1",
            "DIVIDENDOS": "D1",
            "RENDIMENTO": "R1",
            "ACORDO COMERCIAL": "I1",
            "IRRF S/DAY TRADE": "T1",
            "IRRF S/ OPERACOES": "T1",
            "IRRF VENDA RENDA FIXA": "T1",
            "CREDITO FRACOES": "CF",
            "RESGATE DE RENDA VARIAVEL": "RRV",
            "RECEBIMENTO": "C",
            "RETIRADA": "W",
            "VENDA RENDA FIXA": "C",
            "COMPRA RENDA FIXA": "W",
        }

        self.symbol_map = {"IRRF S/DAY TRADE": "TAX", "IRRF S/ OPERACOES": "TAX", "IRRF VENDA RENDA FIXA": "TAX"}

    def concat_files(self):
        files = sorted(glob(self.inputDir + "/*.csv"))
        for file in files:
            tmp = pd.read_csv(file, sep=";", decimal=",", thousands=".")
            self.dtFrame = pd.concat([self.dtFrame, tmp.iloc[::-1]])
        self.dtFrame.columns = "DATE PAYDATE PRICE DESCRIPTION CASH".split()
        self.dtFrame[DataSchema.DATE] = pd.to_datetime(self.dtFrame[DataSchema.DATE], format=DataSchema.DATE_FORMAT)
        self.dtFrame[DataSchema.PAYDATE] = pd.to_datetime(
            self.dtFrame[DataSchema.PAYDATE], format=DataSchema.DATE_FORMAT
        )
        self.dtFrame.to_csv(self.inputDir + "/extrato.tsv", index=False, sep="\t")

    def description_parser(self, value):
        error = False
        op = qty = symbol = np.nan
        replace_dic = {"Ç": "C", "Õ": "O", "Ã": "A", "Á": "A", "É": "E"}
        value = value.strip().upper()
        for src, dest in replace_dic.items():
            value = value.replace(src, dest)

        try:
            while True:
                res = re.compile(r"NOTA.+(CORRETAGEM|PREGAO).*").search(value)
                if res:
                    break

                res = re.compile(
                    r"(RENDIMENTOS?|DIVIDENDOS?|JUROS S\/CAPITAL)\s+([0-9,.]+)\s+(?:PAPEL)?\s(\w+)"
                ).search(value)
                if res:
                    op = self.operation_map[res.group(1)]
                    qty = int(res.group(2).replace(".", "").replace(",", ""))
                    symbol = res.group(3)
                    break

                res = re.compile(
                    r"^(?:TED.*)?(VENDA RENDA FIXA|COMPRA RENDA FIXA|RECEBIMENTO|RETIRADA|IRRF S\/DAY TRADE|IRRF S\/ OPERACOES|IRRF VENDA RENDA FIXA|ACORDO COMERCIAL).*"
                ).search(value)
                if res:
                    op = self.operation_map[res.group(1)]
                    symbol = self.symbol_map.get(res.group(1), DataSchema.CASH)
                    qty = 1
                    break

                res = re.compile(r"(PAGAMENTO DE AMORTIZACAO)\s*(\w+)?").search(value)
                if res:
                    op = self.operation_map[res.group(1)]
                    if res.group(2):
                        symbol = res.group(2)
                    # print(f'value: {value}\t|\tmatch: {res.groups()}\t|\treturn: op={op}, symbol={symbol}, qty={qty}\n')
                    break

                res = re.compile(r"(?:PAGA\w+ DE\s+)?(RESGATE DE RENDA VARIAVEL|CREDITO FRACOES)\s+(\w+)").search(value)
                if res:
                    op = self.operation_map[res.group(1)]
                    symbol = res.group(2)
                    qty = 1
                    break

                print(f"No match for: {value}")
                break
        except Exception as e:
            print(f"Exception: {e}\tWhen parsing line: {value}\n")
            exit()

        return op, qty, symbol

    def process(self):
        self.concat_files()
        self.dtFrame[DataSchema.OPERATION], self.dtFrame[DataSchema.QUANTITY], self.dtFrame[DataSchema.SYMBOL] = zip(
            *self.dtFrame[DataSchema.DESCRIPTION].map(self.description_parser)
        )

        operation_order_map = {"A1": 0, "R1": 1, "D1": 2, "JCP1": 3, "T1": 4, "I1": 5}
        self.dtFrame["OPERATION_ORDER"] = self.dtFrame[DataSchema.OPERATION].map(
            lambda x: operation_order_map.get(x, 10)
        )
        self.dtFrame.sort_values([DataSchema.PAYDATE, "OPERATION_ORDER"], inplace=True)

        self.dtFrame[DataSchema.SYMBOL] = self.dtFrame[DataSchema.SYMBOL].fillna(
            self.dtFrame[DataSchema.SYMBOL].shift(-1)
        )
        self.dtFrame[DataSchema.QUANTITY] = self.dtFrame[DataSchema.QUANTITY].fillna(
            self.dtFrame[DataSchema.QUANTITY].shift(-1)
        )
        self.dtFrame[DataSchema.PRICE] /= self.dtFrame[DataSchema.QUANTITY]
        self.dtFrame = self.dtFrame.dropna()
        self.dtFrame = self.dtFrame["SYMBOL DATE PRICE PAYDATE OPERATION QUANTITY DESCRIPTION".split()]
        self.dtFrame[DataSchema.DATE] = pd.to_datetime(self.dtFrame[DataSchema.DATE], format=DataSchema.DATE_FORMAT)
        self.dtFrame[DataSchema.PAYDATE] = pd.to_datetime(
            self.dtFrame[DataSchema.PAYDATE], format=DataSchema.DATE_FORMAT
        )
        # exit()

    def finish(self):
        self.dtFrame.to_csv(self.output, index=False, sep="\t")
        return self.dtFrame
