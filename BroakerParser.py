import re
import pdfplumber
import numpy as np
from collections import namedtuple
import pandas as pd

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
        self.operation_re = re.compile(r'[\w\d-]+\s(C|V)\s+(?:VISTA|FRACIONARIO)\s(?:\d\d/\d\d)?([\w\d\s./]+?)\s\s+([\w\d\s#]+?)\s(\d+)\s([\d.,]+)\s([\d.,]+)\s(\w)')
        self.liqFee_re = re.compile(r'.*?Taxa de liquida.*?\s+([\d,]+)')
        self.emolFee_re = re.compile(r'Emolumentos\s+([\d,]+)')
        self.opFee_re = re.compile(r'Taxa Operacional\s+([\d,]+)')
        self.exFee_re = re.compile(r'Execu\w+\s+([\d,]+)')
        self.custodyFee_re = re.compile(r'.*?Taxa de Cust\w+\s+([\d,]+)')
        self.irrf_re = re.compile(r'I.R.R.F.*?base.*?[\d,]+\s([\d,]+)')
        self.otherFee_re = re.compile(r'Outros\s+([\d,]+)')
        super().__init__(outDir, name)

    def process(self, page):
        text = page.extract_text()

        order = namedtuple('order', 'Code Date Company Type Category Qty Value Total Sub ')
        line_itens = []
        for line in text.split('\n'):
            res = self.operation_re.search(line)
            if res:
                # print (res.group(0))
                opType = 'Venda' if res.group(1)=='V' else 'Compra'
                name = res.group(2).strip()
                code = res.group(3).strip()

                if (('FII' in name) or ('FDO' in name)):
                    category = 'FII'
                    code = code.split(' ')[0]
                else:
                    category = 'Ação'

                line_itens.append(order(code, Date, name, opType, category, int(res.group(4)), to_float(res.group(5)), to_float(res.group(6)), code.split(' ')[0] ))
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
        df['otherFee'] = otherFee * df['Total'] / total
        df['Fee'] = df['LiqFee'] + df['EmolFee'] + df['OpFee'] + df['ExFee'] + df['CustodyFee'] + df['Irrf'] + df['otherFee']
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
                opType = 'Compra' if res.group(1) == 'BOUGHT' else 'Venda'
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

if __name__ == "__main__":
    pdf = pdfplumber.open('d:/Investing/Notas_TD/Trade_Confirmations.pdf', password='371')
    pgObj = TDAmeritrade('d:/Investing/', 'TD')
    for page in pdf.pages:
        pgObj.process(page)

    pgObj.finish()
