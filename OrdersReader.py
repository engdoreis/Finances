import os
import re
import pandas as pd 
from FinanceTools import *
import pdfplumber
import numpy as np
from collections import namedtuple

# root = '/content/drive/MyDrive'
root = 'd:'
# drive.mount('/content/drive')

companyMap = CompanyListReader().dtFrame
companyMap.to_csv(root + '/Investing/map.csv')

def to_float(str, decimal=',', thousand='.'):
    return float(str.replace(thousand,'').replace(decimal,'.'))

class PDFPage:
    def __init__(self, cmpMap):
        self.date_re = re.compile(r'\d{2}/\d{2}/\d{4}$')
        self.operation_re = re.compile(r'[\w\d-]+\s(C|V)\s+(?:VISTA|FRACIONARIO)\s(?:\d\d/\d\d)?([\w\d\s./]+?)\s\s+([\w\d\s#]+?)\s(\d+)\s([\d.,]+)\s([\d.,]+)\s(\w)')
        self.liqFee_re = re.compile(r'.*?Taxa de liquida.*?\s+([\d,]+)')
        self.emolFee_re = re.compile(r'Emolumentos\s+([\d,]+)')
        self.opFee_re = re.compile(r'Taxa Operacional\s+([\d,]+)')
        self.exFee_re = re.compile(r'Execu\w+\s+([\d,]+)')
        self.custodyFee_re = re.compile(r'.*?Taxa de Cust\w+\s+([\d,]+)')
        self.irrf_re = re.compile(r'I.R.R.F.*?base.*?[\d,]+\s([\d,]+)')
        self.otherFee_re = re.compile(r'Outros\s+([\d,]+)')
        self.dtFrame = pd.DataFrame(columns=['order', 'Code', 'Date', 'Company', 'Type', 'Category', 'Qty', 'Value', 'Total', 'Sub'])
        self.cmpMap = cmpMap    

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
                    category = 'Stock'          

                line_itens.append(order(code, Date, name, opType, category, int(res.group(4)), to_float(res.group(5)), to_float(res.group(6)), code.split(' ')[0] ))
                continue

            res = self.date_re.search(line)
            if res:
                Date = res.group(0)
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

    def partialMatch(self, row):
        if ('FII' in row['Category']):
            row['Paper'] = row['Code']
        elif (str(row['Paper']) == 'nan'):
            filter = self.cmpMap['Company'].str.contains(row.Company)  
            if(not filter.any()):
                filter = self.cmpMap['FullName'].str.contains(row.Company)
            if(not filter.any()):
                filter = self.cmpMap['Company'].isin(row.Company.split(' '))

            df = self.cmpMap[filter & (self.cmpMap.Sub == row.Sub)]

            if(not df.empty):
                row['Paper'] = df.iloc[0]['Paper']
                
        return row

    def finish(self):
        self.dtFrame = self.dtFrame.merge(self.cmpMap, how='left', on=['Company', 'Sub'])
        self.dtFrame = self.dtFrame.apply(self.partialMatch, axis=1).reset_index(drop=True)
        pageObj.dtFrame['Date'] = pd.to_datetime(pageObj.dtFrame['Date'], format='%d/%m/%Y')
        self.dtFrame = self.dtFrame.sort_values('Date').reset_index(drop=True)
        return self.dtFrame

pdf = None
try:
    pdf = pdfplumber.open(root+'/Investing/Notas_Clear/161936_NotaCorretagem.pdf')
except:
    print('File not exist')
    if (pdf):
        pageObj = PDFPage(companyMap)
    for idx, page in enumerate(pdf.pages[0:]):
        # print(f'Page {idx}')
        pageObj.process(page)

    pageObj.finish()

from tqdm import tqdm
from glob import glob
import time

directory = root + 'Investing/Notas_Clear'
pageObj = PDFPage(companyMap)
files = sorted(glob(directory + '/*.pdf'))

for file in tqdm(files, ncols=100, colour='green'):
    # print(f'Processing {file}')
    pdf = pdfplumber.open(file)
    for page in pdf.pages:
        pageObj.process(page)

pageObj.finish().head(2)

#Gambiarra 1 UNT -> 3 ON
pageObj.dtFrame.loc[pageObj.dtFrame.Paper == 'VVAR11','Qty'] *= 3
pageObj.dtFrame.loc[pageObj.dtFrame.Paper == 'VVAR11','Value'] /= 3
pageObj.dtFrame.loc[pageObj.dtFrame.Paper == 'VVAR11','Sub'] = 'ON'
pageObj.dtFrame.loc[pageObj.dtFrame.Paper == 'VVAR11','Paper'] = 'VVAR3'


pageObj.dtFrame.loc['Date'] = pageObj.dtFrame['Date'].dt.strftime('%d-%m-%Y')
pageObj.dtFrame[['Paper', 'Date', 'Value', 'Qty', 'Type', 'Category', 'Fee', 'Company']].to_csv(root+'/Investing/operations_2.csv')


