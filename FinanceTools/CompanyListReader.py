import pandas as pd
import requests

http_header = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.75 Safari/537.36",
    "X-Requested-With": "XMLHttpRequest",
}


class CompanyListReader:
    def __init__(self):
        self.dtFrame = self.loadBmfBovespa()

    def loadInfomoney(self):
        url = "https://www.infomoney.com.br/minhas-financas/confira-o-cnpj-das-acoes-negociadas-em-bolsa-e-saiba-como-declarar-no-imposto-de-renda/"
        r = requests.get(url, headers=http_header)
        rawTable = pd.read_html(r.text, thousands=".", decimal=",")[0]
        return rawTable

    def loadBmfBovespa(self):
        url = "https://bvmf.bmfbovespa.com.br/CapitalSocial/"
        try:
            r = requests.get(url, headers=http_header, timeout=5)
        except:
            print(f"Error to read url: {url}")
            return pd.DataFrame()

        rawTable = pd.read_html(r.text, thousands=".", decimal=",")[0]
        rawTable = rawTable.iloc[:, :4]  # Remove columns after 4th column.
        rawTable.columns = ["NAME", "CODE", "SOCIAL_NAME", "SEGMENT"]
        return rawTable

    def loadOceans(self):
        url = "https://www.oceans14.com.br/acoes/"
        r = requests.get(url, headers=http_header)
        rawTable = pd.read_html(r.text, thousands=".", decimal=",")[0]
        return rawTable

    def loadGuiaInvest(self):
        pageAmount = 10
        rawTable = pd.DataFrame()
        url = "https://www.guiainvest.com.br/lista-acoes/default.aspx?listaacaopage="
        r = requests.get(url, headers=http_header)
        df = pd.read_html(r.text, thousands=".", decimal=",")[0]
        res = re.search("Registros\s\d+\s-\s(\d+)\sde\s(\d+)", df.to_string())
        if res:
            pageAmount = ceil(int(res.group(2)) / int(res.group(1)))

        for i in range(1, pageAmount):
            r = requests.get(url + str(i), headers=http_header)
            rawTable = pd.concat(
                [
                    rawTable,
                    pd.read_html(r.text, thousands=".", decimal=",")[0].drop(
                        ["Unnamed: 0", "Atividade Principal"], axis=1
                    ),
                ]
            )

        return rawTable.reset_index(drop=True)

    def loadAdvfn(self):
        rawTable = pd.DataFrame()
        url = "https://br.advfn.com/bolsa-de-valores/bovespa/"
        for pg in string.ascii_uppercase:
            r = requests.get(url + pg, headers=http_header)
            # rawTable = rawTable.append(pd.read_html(r.text, thousands='.',decimal=',')[0])
            rawTable = pd.concat([rawTable, pd.read_html(r.text, thousands=".", decimal=",")[0]])

        return rawTable[["Ação", "Unnamed: 1"]].dropna()

    def loadFundamentus(self):
        def SubCategory(row):
            if "3" in row["Paper"]:
                row["Sub"] = "ON"
            elif "4" in row["Paper"]:
                row["Sub"] = "PN"
            else:
                row["Sub"] = "UNT"
            return row

        url = "https://www.fundamentus.com.br/detalhes.php?papel="
        r = requests.get(url, headers=http_header)
        rawTable = pd.read_html(r.text, thousands=".", decimal=",")[0].fillna("Unown")
        rawTable.columns = ["Paper", "Company", "FullName"]
        rawTable = rawTable.apply(lambda x: x.str.upper())

        rawTable = rawTable.apply(SubCategory, axis=1)
        return rawTable
