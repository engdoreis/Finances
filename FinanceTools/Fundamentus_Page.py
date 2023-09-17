import pandas as pd

from data import DataSchema

http_header = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.75 Safari/537.36",
    "X-Requested-With": "XMLHttpRequest",
}


class Fundamentus_Page:
    urlDict = {
        "AÇÃO": "https://www.fundamentus.com.br/proventos.php?papel={}&tipo=2",
        "FII": "https://www.fundamentus.com.br/fii_proventos.php?papel={}&tipo=2",
    }

    def __init__(self, type):
        self.url = self.urlDict[type.upper()]

    def read(self, ticker):
        res = pd.DataFrame()
        url = self.url.format(ticker)
        r = requests.get(url, headers=http_header)
        try:
            rawTable = pd.read_html(r.text, thousands=".", decimal=",")[0]
            if not "Valor" in rawTable.columns:
                raise
        except:
            return res

        res = rawTable
        if "Por quantas ações" in res.columns:
            res["Valor"] /= res["Por quantas ações"]

        if "Última Data Com" in res.columns:
            res.rename(columns={"Última Data Com": "Data"}, inplace=True)

        res.rename(
            columns={
                "Tipo": DataSchema.OPERATION,
                "Data": DataSchema.DATE,
                "Data de Pagamento": DataSchema.PAYDATE,
                "Valor": DataSchema.PRICE,
                "Tipo": DataSchema.OPERATION,
            },
            inplace=True,
        )
        operation_map = {
            "AMORTIZAÇÃO": "A",
            "JRS CAP PROPRIO": "JCP",
            "DIVIDENDO": "D",
            "RENDIMENTO": "D",
            "DIVIDENDO MENSAL": "D",
            "JUROS": "JCP",
            "JRS CAP PRÓPRIO": "JCP",
            "JUROS MENSAL": "JCP",
        }
        res[DataSchema.OPERATION] = res[DataSchema.OPERATION].map(lambda x: operation_map[x.upper()])

        return res
