import os, sys
import re
from shutil import rmtree
import pandas as pd 
from FinanceTools import *
import pdfplumber
import numpy as np
from collections import namedtuple
from multiprocessing import Process
from threading import Thread
from glob import glob
import time

def to_float(str, decimal=',', thousand='.'):
    return float(str.replace(thousand,'').replace(decimal,'.'))

class PDFPage:
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
        self.output = outDir + '/' + name + '.csv'
        self.dtFrame = pd.DataFrame(columns=['Code', 'Date', 'Company', 'Type', 'Category', 'Qty', 'Value', 'Total', 'Sub'])
        
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

    def finish(self):
        self.dtFrame.to_csv(self.output, index=False, float_format='%.5f')

class OrderOrganizer:
    def __init__(self, inDir):
        self.dtFrame = pd.DataFrame(columns=['Code', 'Date', 'Company', 'Type', 'Category', 'Qty', 'Value', 'Total', 'Sub'])

        files = sorted(glob(inDir + '/*.csv'))
        for file in files:
            self.dtFrame = self.dtFrame.merge(pd.read_csv(file), how='outer')

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

    def finish(self, cmpMap):
        self.cmpMap = cmpMap
        self.dtFrame = self.dtFrame.merge(self.cmpMap, how='left', on=['Company', 'Sub'])
        self.dtFrame = self.dtFrame.apply(self.partialMatch, axis=1).reset_index(drop=True)
        self.dtFrame['Date'] = pd.to_datetime(self.dtFrame['Date'], format='%d/%m/%Y')
        self.dtFrame = self.dtFrame.sort_values('Date').reset_index(drop=True)
        return self.dtFrame

def ReadPages(file, dir_):
    pdf = pdfplumber.open(file, password='371')
    pgObj = PDFPage(dir_, os.path.basename(file).split('.')[0])
    for page in pdf.pages:
        pgObj.process(page)

    pgObj.finish()

def ReadOrders(indir='d:/Investing/Notas_Clear', outfile='operations.csv'):
    inputDir = indir
    outputDir = indir + '/..'
    tmpDir = outputDir + '/tmpDir'

    try:
        rmtree(tmpDir)
    finally:
        os.mkdir(tmpDir)

    # pageObj = PDFPage()
    files = sorted(glob(inputDir + '/*.pdf'))

    processes = []

    print('Starting pages')
    for file in files:
        processes.append(Process(target=ReadPages, args=(file,tmpDir)))

    for pcs in processes:
        pcs.start()

    print('Getting tickers names...', end='\r')

    companyListReader = CompanyListReader()
    print('Getting tickers names...Done')

    for pcs in processes:
        pcs.join()

    print('Pages done')
    print('Tickers merging...', end='\r')
    companyMap = companyListReader.dtFrame
    companyMap.to_csv(outputDir + '/map.csv')

    oOrg = OrderOrganizer(tmpDir)
    oOrg.finish(companyMap)

    print('Tickers merging...Done')
    #Gambiarra 1 UNT -> 3 ON
    oOrg.dtFrame.loc[oOrg.dtFrame.Paper == 'VVAR11','Qty'] *= 3
    oOrg.dtFrame.loc[oOrg.dtFrame.Paper == 'VVAR11','Value'] /= 3
    oOrg.dtFrame.loc[oOrg.dtFrame.Paper == 'VVAR11','Sub'] = 'ON'
    oOrg.dtFrame.loc[oOrg.dtFrame.Paper == 'VVAR11','Paper'] = 'VVAR3'
    oOrg.dtFrame.loc['Date'] = oOrg.dtFrame['Date'].dt.strftime('%d-%m-%Y')

    tempFile = outputDir + '/tmp_' + outfile
    outPath = outputDir + '/' + outfile
    oOrg.dtFrame[['Paper', 'Date', 'Value', 'Qty', 'Type', 'Category', 'Fee', 'Company']].to_csv(tempFile, index=False)

    try:
        existentDF = pd.read_csv(outPath)
        outDF = pd.read_csv(tempFile)
        existentDF=existentDF[existentDF['Date'].astype(bool)].dropna()

        diff = outDF.merge(existentDF, how='outer', on=['Date', 'Value', 'Qty', 'Fee', 'Company'], suffixes=['','_'], indicator=True)
        diff = diff[diff['_merge'] != 'both']
        diff = diff.iloc[:,:8]
        
        existentDF.append(diff).to_csv(outPath, index=False)
        os.remove(tempFile)
    except:
        os.rename(tempFile, outPath)
        

if __name__ == "__main__":
    # start_time = time.time()
    indir='d:/Investing/Notas_Clear'
    outfile='operations.csv'
    if(len(sys.argv) > 2):
        indir = str(sys.argv[1])
        outfile = str(sys.argv[2])

    ReadOrders(indir, outfile)

    # print("--- %s seconds ---" % (time.time() - start_time))


