import os, sys
import re
from shutil import rmtree
import pandas as pd 
import pdfplumber
import numpy as np
from multiprocessing import Process
from threading import Thread
from glob import glob
import time
from FinanceTools import *
from BroakerParser import *

class OrderOrganizer:
    def __init__(self, inDir):
        self.dtFrame = pd.DataFrame(columns=['Code', 'Date', 'Company', 'Type', 'Category', 'Qty', 'Value', 'Total', 'Sub'])

        files = sorted(glob(inDir + '/*.csv'))
        for file in files:
            self.dtFrame = self.dtFrame.merge(pd.read_csv(file), how='outer')

    def partialMatch(self, row):
        code = row['Code']
        if ('FII' in row['Category']):
            row['Paper'] = code
        else:
            code_number = 0
            if 'PN' in code:
                code_number = 4
            elif 'ON' in code:
                code_number = 3
            elif 'UNT' in code:
                code_number = 11
            else:
                raise f'Unknown stock type: {code}'

            row['Paper'] = row['CODE'] + str(code_number)
                
        return row

    def finish(self, cmpMap):
        if(self.dtFrame['Code'].str.contains('ON|PN|UNT').any()):
            self.cmpMap = cmpMap
            self.dtFrame = self.dtFrame.merge(self.cmpMap, how='left', left_on='Company', right_on='NAME')
            self.dtFrame = self.dtFrame.apply(self.partialMatch, axis=1).reset_index(drop=True)
            self.dtFrame['Date'] = pd.to_datetime(self.dtFrame['Date'])
            self.dtFrame = self.dtFrame.sort_values('Date').reset_index(drop=True)
        else:
            self.dtFrame['Paper'] = self.dtFrame['Code']
        return self.dtFrame

def ReadPages(file, dir_, pdfType='Clear'):
    pdf = pdfplumber.open(file, password='371')
    
    if(pdfType == 'Clear'):
        pgObj = Clear(dir_, os.path.basename(file).split('.')[0])
    else:
        pgObj = TDAmeritrade(dir_, os.path.basename(file).split('.')[0])

    for page in pdf.pages:
        pgObj.process(page)

    pgObj.finish()

def ReadOrders(indir='d:/Investing/Notas_Clear', outfile='d:/Investing/operations.csv', pdfType='Clear'):
    inputDir = indir
    outputDir = indir + '/..'
    tmpDir = outputDir + '/tmpDir'

    if os.path.exists(tmpDir):
        rmtree(tmpDir)
    os.mkdir(tmpDir)

    # pageObj = PDFPage()
    files = sorted(glob(inputDir + '/*.pdf'))

    processes = []

    print('Starting pages')
    for file in files:
        processes.append(Process(target=ReadPages, args=(file,tmpDir, pdfType, )))

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
    
    tempFile = outfile + '.tmp'
    oOrg.dtFrame[['Paper', 'Date', 'Value', 'Qty', 'Type', 'Category', 'Fee', 'Company']].to_csv(tempFile, index=False)

    try:
        dtypes =  {'Qty': float, 'Value': float}
        existentDF = pd.read_csv(outfile, dtype = dtypes)
        outDF = pd.read_csv(tempFile, dtype = dtypes)
        existentDF=existentDF[existentDF['Date'].astype(bool)].dropna()

        diff = outDF.merge(existentDF, how='outer', on=['Date', 'Value', 'Qty', 'Fee', 'Company'], suffixes=['','_'], indicator=True)
        diff = diff[diff['_merge'] == 'left_only']
        diff = diff.iloc[:,:8]
        
        # existentDF.append(diff).to_csv(outfile, index=False)
        new = pd.concat([existentDF, diff])
        new['Qty'] = new['Qty'].astype(float).round(6)
        new.to_csv(outfile, index=False)
        os.remove(tempFile)
    except:
        os.rename(tempFile, outfile)

def ReadTDStatement(inDir='d:/Investing/Notas_TD', outfile='d:/Investing/TD.csv'):
    def DescriptionParser(row):
        desc = row['DESCRIPTION']
        if ('Bought' in desc):
            row['DESCRIPTION'] = 'B'
        if ('Sold' in desc):
            row['DESCRIPTION'] = 'S'
        if ('DIVIDEND' in desc):
            row['DESCRIPTION'] = 'D1'
            row['QUANTITY'] = 1
            row['PRICE'] = row['AMOUNT']
        if ('GAIN DISTRIBUTION' in desc):
            row['DESCRIPTION'] = 'D1'
            row['QUANTITY'] = 1
            row['PRICE'] = row['AMOUNT']
        if ('TAX WITHHELD' in desc):
            row['DESCRIPTION'] = 'D1'
            row['QUANTITY'] = 1
            row['PRICE'] = row['AMOUNT']
        if ('W-8' in desc): #Dividend Taxes
            row['DESCRIPTION'] = 'T1'
            row['QUANTITY'] = 1
            row['PRICE'] = row['AMOUNT']
        if ('WIRE' in desc):
            row['DESCRIPTION'] = 'C'
            row['TYPE'] = 'WIRE'
            row['SYMBOL'] = 'CASH'
            row['QUANTITY'] = 1
            row['PRICE'] = row['AMOUNT']
        if ('INTEREST' in desc):
            row['DESCRIPTION'] = 'C'
            row['TYPE'] = 'INTEREST'
            row['SYMBOL'] = 'CASH'
            row['QUANTITY'] = 1
            row['PRICE'] = row['AMOUNT']
        return row
    
    table = pd.DataFrame()
    for file in sorted(glob(inDir + '/*.csv')):
        df = pd.read_csv(file)
        df=df[~df['DATE'].str.contains('END OF FILE')].fillna(0)
        df['TYPE'] = 'STOCK'
        df['DATE'] = pd.to_datetime(df['DATE']).dt.strftime('%Y-%m-%d')
        df = df[['SYMBOL', 'DATE', 'PRICE', 'QUANTITY', 'DESCRIPTION', 'TYPE', 'COMMISSION', 'AMOUNT']]

        df = df.apply(DescriptionParser, axis=1)
        df = df.rename(columns={'DESCRIPTION':'OPERATION'})
        if table.empty:
            table = pd.concat([table, df])
        else:
            table = table.merge(df, how='outer', on=['SYMBOL', 'DATE', 'PRICE', 'QUANTITY', 'OPERATION', 'TYPE', 'COMMISSION', 'AMOUNT'],  suffixes=['','_'], indicator=True)
            table.columns.drop(['_merge'])
            table = table.loc[:,~table.columns.str.endswith('_')]

    table.to_csv(outfile, index=False)

if __name__ == "__main__":
    # start_time = time.time()
    indir='d:/Investing/Notas_Clear'
    outfile='operations.csv'
    pdfType='Clear'
    if(len(sys.argv) > 2):
        indir = str(sys.argv[1])
        outfile = str(sys.argv[2])
        pdfType=str(sys.argv[3])

    ReadOrders(indir, outfile, pdfType)

    # print("--- %s seconds ---" % (time.time() - start_time))


