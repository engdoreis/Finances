import requests
import pandas as pd

from data import DataSchema

http_header = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.75 Safari/537.36",
    "X-Requested-With": "XMLHttpRequest",
}


class ADVFN_Page:
    def find_table(self, df):
        for index in range(len(df)):
            tmp = df[index]
            if "Valor" in tmp.columns:
                return tmp
        return pd.DataFrame()

    def read(self, ticker):
        res = pd.DataFrame()
        url = "https://br.advfn.com/bolsa-de-valores/bovespa/{}/dividendos/historico-de-proventos".format(ticker)
        r = requests.get(url, headers=http_header)
        try:
            rawTable = self.find_table(pd.read_html(r.text, thousands=".", decimal=","))
            if rawTable.empty:
                raise
        except:
            print(f"{ticker} not found at {url}")
            return res

        res = rawTable
        if "Mês de Referência" in res.columns:
            res.rename(columns={"Mês de Referência": "Tipo do Provento"}, inplace=True)
            res["Tipo do Provento"] = "Dividendo"

        res.rename(
            columns={
                "Tipo do Provento": DataSchema.OPERATION,
                "Data-Com": DataSchema.DATE,
                "Pagamento": DataSchema.PAYDATE,
                "Valor": DataSchema.PRICE,
                "Dividend Yield": "YIELD",
            },
            inplace=True,
        )
        operation_map = {
            "AMORTIZAÇÃO": "A",
            "JUROS SOBRE CAPITAL PRÓPRIO": "JCP",
            "DIVIDENDO": "D",
            "RENDIMENTOS": "D",
            "RENDIMENTO": "D",
            "DESDOBRAMENTO": "SPLIT1",
        }
        res[DataSchema.OPERATION] = res[DataSchema.OPERATION].map(lambda x: operation_map[x.upper()])

        return res
