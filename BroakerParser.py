import re
from unicodedata import decimal
import pdfplumber
import numpy as np
from collections import namedtuple
import pandas as pd
from glob import glob

def to_float(str, decimal=',', thousand='.'):
    return float(str.replace(thousand,'').replace(decimal,'.'))

class Broaker():
    def __init__(self, outDir, name='default'):
        self.output = outDir + '/' + name + '.csv'
        self.dtFrame = pd.DataFrame(columns=['Code', 'Date', 'Company', 'Type', 'Category', 'Qty', 'Value', 'Total', 'Sub'])

    def process(self, page):
        pass

    def finish(self):
        self.dtFrame.to_csv(self.output, index=False, float_format='%.5f')

class Clear(Broaker):
    def __init__(self, outDir, name='default'):
        self.date_re = re.compile(r'\d{2}/\d{2}/\d{4}$')
        self.operation_re = re.compile(r'[\w\d-]+\s(C|V)\s+(?:VISTA|FRACIONARIO)\s(?:\d\d\/\d\d)?([\w\d\s.\/]+?)\s\s+([\w\d\s#]+?)\s([\d.,]+)\s([\d.,]+)\s([\d.,]+)\s(\w)')
        self.liqFee_re = re.compile(r'.*?Taxa de liquida.*?\s+([\d,]+)')
        self.emolFee_re = re.compile(r'Emolumentos\s+([\d,]+)')
        self.opFee_re = re.compile(r'Taxa Operacional\s+([\d,]+)')
        self.exFee_re = re.compile(r'Execu\w+\s+([\d,]+)')
        self.custodyFee_re = re.compile(r'.*?Taxa de Cust\w+\s+([\d,]+)')
        self.irrf_re = re.compile(r'I.R.R.F.*?base.*?[\d,]+\s([\d,]+)')
        self.taxes_re = re.compile(r'Impostos\s+([\d,]+)')
        self.otherFee_re = re.compile(r'Outros\s+([\d,]+)')
        super().__init__(outDir, name)

    def process(self, page):
        liqFee = liqFee = emolFee = opFee = exFee = custodyFee = irrf = taxes = otherFee = 0

        text = page.extract_text()
        order = namedtuple('order', 'Code Date Company Type Category Qty Value Total Sub ')
        line_itens = []
        for line in text.split('\n'):
            res = self.operation_re.search(line)
            if res:
                # print (res.group(0))
                opType = 'S' if res.group(1)=='V' else 'B'
                name = res.group(2).strip()
                code = res.group(3).strip()

                if (('FII' in name) or ('FDO' in name)):
                    category = 'FII'
                    code = code.split(' ')[0]
                else:
                    category = 'Ação'

                line_itens.append(order(code, Date, name, opType, category, int(res.group(4).replace('.','')), to_float(res.group(5)), to_float(res.group(6)), code.split(' ')[0] ))
                continue

            res = self.date_re.search(line)
            if res:
                Date = pd.to_datetime(res.group(0), format='%d/%m/%Y').strftime('%Y-%m-%d')
                continue

            res = self.liqFee_re.search(line)
            if res:
                liqFee = to_float(res.group(1))
                continue

            res = self.emolFee_re.search(line)
            if res:
                emolFee =to_float( res.group(1))
                continue
            
            res = self.opFee_re.search(line)
            if res:
                opFee =to_float( res.group(1))
                continue
            
            res = self.exFee_re.search(line)
            if res:
                exFee =to_float( res.group(1))
                continue

            res = self.custodyFee_re.search(line)
            if res:
                custodyFee =to_float( res.group(1))
                continue

            res = self.irrf_re.search(line)
            if res:
                irrf = to_float( res.group(1))
                continue

            res = self.taxes_re.search(line)
            if res:
                taxes = to_float( res.group(1))
                continue

            res = self.otherFee_re.search(line)
            if res:
                otherFee = to_float( res.group(1))
                continue

        df = pd.DataFrame(line_itens)

        total = df['Total'].sum()
        df['LiqFee'] = liqFee * df['Total'] / total
        df['EmolFee'] = emolFee * df['Total'] / total
        df['OpFee'] = opFee * df['Total'] / total
        df['ExFee'] = exFee * df['Total'] / total
        df['CustodyFee'] = custodyFee * df['Total'] / total
        df['Irrf'] = irrf * df['Total'] / total
        df['Taxes'] = taxes * df['Total'] / total
        df['otherFee'] = otherFee * df['Total'] / total
        df['Fee'] = df['LiqFee'] + df['EmolFee'] + df['OpFee'] + df['ExFee'] + df['CustodyFee'] + df['Taxes'] + df['otherFee']
        self.dtFrame = self.dtFrame.merge(df, how='outer')

class TDAmeritrade(Broaker):
    def __init__(self, outDir, name='default'):
        self.output = outDir + '/' + name + '.csv'
        self.operation_re = re.compile(r'YOU\s(BOUGHT|SOLD)\s+(\d+)\s+.+?\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)')
        self.date_re = re.compile(r'(\d{2}\/\d{2}\/\d{4})\s+(\d{2}\/\d{2}\/\d{4})\s+([\d.]+)\s+([\d.]+)')
        self.Ticker_re = re.compile(r'^\s(\w+)\s\s\w+(\s\w+)?$')
        super().__init__(outDir, name)
        
    def process(self, page):
        text = page.extract_text()

        order = namedtuple('order', 'Code Date Company Type Category Qty Value Total Sub Fee')
        line_itens = []
        for line in text.split('\n'):
            res = self.operation_re.search(line)
            if res:
                # print (res.group(0))
                opType = 'B' if res.group(1) == 'BOUGHT' else 'S'
                qty = int(res.group(2))
                value = float(res.group(3))
                fee = float(res.group(5))
                continue

            res = self.date_re.search(line)
            if res:
                # print (res.group(0))
                date = pd.to_datetime(res.group(1), format='%m/%d/%Y').strftime('%Y-%m-%d')
                total = res.group(4)
                continue

            res = self.Ticker_re.search(line)
            if res:
                # print (res.group(0))
                line_itens.append(order(res.group(1), date, 'Company', opType, 'Stock', qty, value, total,'sub',fee))
                continue
        self.dtFrame = self.dtFrame.merge(pd.DataFrame(line_itens), how='outer')
class Clear_proventos():
    def __init__(self, input, outDir, name='default'):
        self.input = input
        self.output = outDir + '/' + name + '.csv'
        self.dtFrame = pd.DataFrame(columns=list('Ativo Evento Quantidade ValorBruto ValorIR ValorLiquido Pagamento'.split()))
        self.dividend_re = re.compile(r'DIVIDENDOS\s([\d]+)\s')

        self.single_evt = re.compile(r'([a-zA-Z/. ]+)\s([a-zA-Z/]+)\s([0-9,.]+)\s([0-9,.]+)\s([0-9,.]+)\s([0-9,.]+)\s([0-9/]+)')
        
    def process(self, page):
        text = page.extract_text()

        order = namedtuple('order', 'Ativo Evento Quantidade ValorBruto ValorIR ValorLiquido Pagamento')
        line_itens = []
        for line in text.split('\n'):
            res = self.single_evt.search(line)
            if res:
                # print (res.group(0))
                line_itens.append(order(res.group(1),res.group(2), res.group(3), res.group(4), res.group(5), res.group(6), res.group(7)))
                continue
            # print(line)

        if len(line_itens) > 0:
          self.dtFrame = self.dtFrame.merge(pd.DataFrame(line_itens), how='outer')

    def finish(self):
        self.dtFrame.to_csv(self.output, index=False, float_format='%.5f')

class Clear_DivStatement():
    def __init__(self, inputDir, outDir, name='default'):
        self.inputDir = inputDir
        self.output = outDir + '/' + name + '.tsv'
        self.dtFrame = pd.DataFrame()
        self.operation_map = {'PAGAMENTO DE AMORTIZAÇÃO': 'A1', 'JUROS-S/CAPITAL':'JCP1', 'DIVIDENDOS': 'D1', 'RENDIMENTO': 'R1', 'ACORDO COMERCIAL': 'I1', 'IRRF S/DAY TRADE': 'T1'}
        
    def concat_files(self):
        files = sorted(glob(self.inputDir + '/*.csv'))
        for file in files:
            tmp =  pd.read_csv(file, sep=';', decimal=',', thousands='.')            
            self.dtFrame = pd.concat([self.dtFrame, tmp.iloc[::-1]])
        self.dtFrame.columns = 'DATE PAYDATE PRICE DESCRIPTION CASH'.split()
        self.dtFrame['DATE'] = pd.to_datetime(self.dtFrame['DATE'], format='%d/%m/%Y')
        self.dtFrame['PAYDATE'] = pd.to_datetime(self.dtFrame['PAYDATE'], format='%d/%m/%Y')
        self.dtFrame.to_csv(self.output + '.tmp', index=False, sep='\t')


    def description_parser(self, value):
        op = qty = symbol = np.nan
        spl_val = value.replace('JUROS S/CAPITAL', 'JUROS-S/CAPITAL').split()
        try:
            if bool(set(spl_val) & set(self.operation_map.keys())):
                op = self.operation_map[spl_val[0]]
                qty = int(spl_val[1].replace('.',''))
                symbol = spl_val[-1]
                raise
            if 'PAGAMENTO DE AMORTIZAÇÃO' in value:
                op = self.operation_map['PAGAMENTO DE AMORTIZAÇÃO']
                symbol = spl_val[-1]
                symbol = np.nan if symbol == 'AMORTIZAÇÃO' else symbol
                raise
            if 'ACORDO COMERCIAL' in value:
                op = self.operation_map['ACORDO COMERCIAL']
                qty = 1
                symbol = 'CASH'
                raise
            if 'IRRF S/DAY TRADE' in value:
                op = self.operation_map['IRRF S/DAY TRADE']
                qty = 1
                symbol = 'TAX'
                raise
            if 'NOTA' in value:
                raise
            if 'TED' in value:
                raise
            print(value)
        except:
            pass

        return op, qty, symbol
        
    def process(self):
        self.concat_files()
        self.dtFrame['OPERATION'], self.dtFrame['QUANTITY'], self.dtFrame['SYMBOL'] = zip(*self.dtFrame['DESCRIPTION'].map(self.description_parser))

        operation_order_map = {'A1': 0, 'R1': 1,  'D1': 2, 'JCP1': 3, 'T1': 4, 'I1': 5}
        self.dtFrame['OPERATION_ORDER'] = self.dtFrame['OPERATION'].map(lambda x: operation_order_map.get(x, 10))
        self.dtFrame.sort_values(['PAYDATE', 'OPERATION_ORDER'], inplace=True)

        self.dtFrame['SYMBOL'] = self.dtFrame['SYMBOL'].fillna(self.dtFrame['SYMBOL'].shift(-1))
        self.dtFrame['QUANTITY'] = self.dtFrame['QUANTITY'].fillna(self.dtFrame['QUANTITY'].shift(-1))
        self.dtFrame['PRICE'] /= self.dtFrame['QUANTITY']
        self.dtFrame = self.dtFrame.dropna()
        self.dtFrame = self.dtFrame['SYMBOL DATE PRICE PAYDATE OPERATION QUANTITY DESCRIPTION'.split()]
        self.dtFrame['DATE'] = pd.to_datetime(self.dtFrame['DATE'], format='%Y-%m-%d')
        self.dtFrame['PAYDATE'] = pd.to_datetime(self.dtFrame['PAYDATE'], format='%Y-%m-%d')

    def finish(self):
        self.dtFrame.to_csv(self.output, index=False, sep='\t')
        return self.dtFrame


def Clear_DivStatementTest():
    clear = Clear_DivStatement('/tmp/clear/', '/tmp/clear/', 'proventos')
    clear.process()
    clear.finish()


def TDAmeritradeTest():
    pdf = pdfplumber.open('d:/Investing/Notas_TD/Trade_Confirmations.pdf', password='371')
    pgObj = TDAmeritrade('d:/Investing/', 'TD')
    for page in pdf.pages:
        pgObj.process(page)

    pgObj.finish()

def ClearProventosTest():
    pdf = pdfplumber.open('/home/doreis/Documents/IRPF/2022/2022_informes_redimento/clear/37165263802-2021-Proventos.pdf', password='371')
    pgObj = Clear_proventos('/home/doreis/Documents/IRPF/2022/2022_informes_redimento/clear/', '/home/doreis/Documents/IRPF/2022/2022_informes_redimento/clear/', 'proventos')
    for page in pdf.pages:
        pgObj.process(page)

    pgObj.finish()

if __name__ == "__main__":
    # TDAmeritradeTest()
    # ClearProventosTest()
    Clear_DivStatementTest()

