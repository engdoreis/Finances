from curses import raw
from socket import timeout
from unicodedata import decimal
from pandas_datareader import data as web
import pandas as pd
import datetime as dt
import numpy as np
from pyparsing import col
from Caching import *
import requests
http_header = {
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.75 Safari/537.36",
        "X-Requested-With": "XMLHttpRequest"
    }

class DataFrameMerger():
    def __init__(self, df):
        self.df = df
    def append(self, new, on):
        columns = self.df.columns.to_list()
        # print(self.df.sort_values(by=['SYMBOL', 'DATE']).reset_index(drop=True))
        # print(new.sort_values(by=['SYMBOL', 'DATE']).reset_index(drop=True))
        self.df = self.df.merge(new, how='outer', on=on, suffixes=['','_'], indicator=True)
        # print(self.df)
        right_cols = [col for col in self.df.columns if col.endswith('_')]
        for col in right_cols:
            self.df[col[:-1]] = self.df[col[:-1]].fillna(self.df[col])
        self.df = self.df[columns]
        self.df = self.df.drop_duplicates()
        # print(self.df.sort_values(by=['SYMBOL', 'DATE']).reset_index(drop=True))
        return self.df
class StockInfoCache():
    def __init__(self, cache_file):
        self.cache = Caching(cache_file)
        self.load_data()

    def is_updated(self):
        # return False
        return self.cache.is_updated()

    def load_data(self):

        table = self.cache.get_data()
        if not table.empty:
            table['DATE'] = pd.to_datetime(table['DATE'], format='%Y-%m-%d')
        self.table = table
        return self.table

    def append(self, data):
        self.cache.append(data)
    
    def merge(self, data, on=['SYMBOL', 'DATE'], sortby=['DATE']):
        if not self.table.empty:
            merger = DataFrameMerger(self.table)
            data = merger.append(data, on=on)
        data = data.sort_values(by=sortby)
        self.cache.save(data)
        return self.load_data()

#     -------------------------------------------------------------------------------------------------

class PriceReader:
    def __init__(self, brTickerList, usTickerList, startDate='2018-01-01'):
        self.brTickerList = brTickerList
        self.usTickerList = usTickerList
        self.startDate = startDate
        self.fillDate = dt.datetime.today().strftime('%m-%d-%Y')
        self.df = pd.DataFrame(columns=['Date'])

    def load(self):
        # Read BR market data
        if((self.brTickerList != None) and (len(self.brTickerList) > 0)):
            self.df = self.readData(self.brTickerList, self.startDate).reset_index()
            self.df.columns = self.df.columns.str.removesuffix('.SA')

        # Read US Market data
        if((self.usTickerList != None) and (len(self.usTickerList) > 0)):
            self.df = self.df.merge(self.readUSData(self.usTickerList, self.startDate).reset_index(), how='outer', on='Date')

        self.df = self.df.set_index('Date').sort_index()
        # self.df.to_csv('debug.csv', sep='\t')

        indexList = ['^BVSP', '^GSPC', 'BRLUSD=X']
        self.brlIndex = self.readUSData(indexList, self.startDate).reset_index()
        self.brlIndex.rename(columns={'^BVSP':'IBOV', '^GSPC':'S&P500', 'BRLUSD=X':'USD'}, inplace=True)
        self.brlIndex = self.brlIndex.merge(self.read_br_selic(self.startDate), on='Date')
        self.brlIndex = self.brlIndex.set_index('Date')

    def setFillDate(self, date):
        self.fillDate = date

    def fillCurrentValue(self, row):
        row['PRICE'] = self.getCurrentValue(row['SYMBOL'], self.fillDate)
        return row
    
    def readData(self, code, startDate='2018-01-01'):
        s=''
        for c in code:
            s += c + '.SA '

        tks = yf.Tickers(s)
        dfs = tks.history(start=startDate, timeout=1000)[['Close']]
        dfs.columns = dfs.columns.droplevel()
        return dfs

    def readUSData(self, code, startDate='2018-01-01'):
        s=''
        for c in code:
            s += c + ' '

        tks = yf.Tickers(s)
        dfs = tks.history(start=startDate)[['Close']]
        dfs.columns = dfs.columns.droplevel()
        return dfs
    
    def read_br_selic(self, startDate='2018-01-01'):
        from bcb import sgs
        selic = sgs.get({'selic':432}, start = startDate)
        selic['selic'] /= 100
        return selic

    def getHistory(self, code, start='2018-01-01'):
        return self.df.loc[start:][code]

    def getCurrentValue(self, code, date=None):
        if not code in self.df:
            return np.nan

        if(date == None):
            return self.df.iloc[-1][code]

        available, date = self.checkLastAvailable(self.df, date, code)
        if available:
            return self.df.loc[date][code] 
        return self.df.iloc[0][code] 

    def getIndexHistory(self, code, end):
        ret = self.brlIndex.loc[:end][code]
        return ret.dropna()

    def getIndexCurrentValue(self, code, date=None):
        if(date == None):
            return self.brlIndex.iloc[-1][code]

        available,date = self.checkLastAvailable(self.brlIndex, date, code)
        if available:
            return self.brlIndex.loc[date][code]
        return self.brlIndex.iloc[0][code]

    def checkLastAvailable(self, dtframe, loockDate, field):
        date = pd.to_datetime(loockDate)
        day = pd.Timedelta(1, unit='d')
        #Look for last available date

        while((not (date in dtframe.index)) or pd.isna(dtframe.loc[date][field])):
            date = date - day
            if(date < dtframe.index[0]):
                return False,0
        return True,date

#     -------------------------------------------------------------------------------------------------

class ADVFN_Page:
    def find_table(self, df):
        for index in range(len(df)):
            tmp = df[index]
            if 'Valor' in tmp.columns:
               return tmp
        return pd.DataFrame()

    def read(self, ticker):
        res = pd.DataFrame()
        url = 'https://br.advfn.com/bolsa-de-valores/bovespa/{}/dividendos/historico-de-proventos'.format(ticker)
        r = requests.get(url, headers=http_header)
        try:
            rawTable = self.find_table(pd.read_html(r.text, thousands='.',decimal=','))
            if rawTable.empty:
                raise
        except:
            print(f'{ticker} not found at {url}')
            return res

        res = rawTable
        if ('Mês de Referência' in res.columns):
            res.rename(columns={'Mês de Referência':'Tipo do Provento'}, inplace=True)
            res['Tipo do Provento'] = 'Dividendo'
        
        res.rename(columns={'Tipo do Provento':'OPERATION', 'Data-Com':'DATE', 'Pagamento':'PAYDATE', 'Valor':'PRICE', 'Dividend Yield':'YIELD'}, inplace=True)
        operation_map = {'AMORTIZAÇÃO': 'A', 'JUROS SOBRE CAPITAL PRÓPRIO':'JCP', 'DIVIDENDO': 'D', 'RENDIMENTOS': 'D', 'RENDIMENTO': 'D', 'DESDOBRAMENTO': 'SPLIT1'}
        res['OPERATION'] = res['OPERATION'].map(lambda x : operation_map[x.upper()])

        return res

class Fundamentus_Page:
    urlDict =  { 'AÇÃO': 'https://www.fundamentus.com.br/proventos.php?papel={}&tipo=2', 
                'FII': 'https://www.fundamentus.com.br/fii_proventos.php?papel={}&tipo=2'}
    def __init__(self, type):
        self.url = self.urlDict[type.upper()]
    
    def read(self, ticker):
        res = pd.DataFrame()
        url = self.url.format(ticker)
        r = requests.get(url, headers=http_header)
        try:
            rawTable = pd.read_html(r.text, thousands='.',decimal=',')[0]
            if not 'Valor' in rawTable.columns:
                raise
        except:
            return res

        res = rawTable
        if('Por quantas ações' in res.columns):
            res['Valor'] /= res['Por quantas ações']

        if ('Última Data Com' in res.columns):
            res.rename(columns={'Última Data Com':'Data'}, inplace=True)

        res.rename(columns={'Tipo':'OPERATION', 'Data':'DATE', 'Data de Pagamento':'PAYDATE', 'Valor':'PRICE', 'Tipo':'OPERATION'}, inplace=True)
        operation_map = {'AMORTIZAÇÃO': 'A', 'JRS CAP PROPRIO':'JCP', 'DIVIDENDO': 'D', 'RENDIMENTO': 'D', 'DIVIDENDO MENSAL': 'D', 'JUROS':'JCP', 'JRS CAP PRÓPRIO':'JCP', 'JUROS MENSAL':'JCP'}
        res['OPERATION'] = res['OPERATION'].map(lambda x : operation_map[x.upper()])

        return res
    
class DividendReader:    
    def __init__(self, brTickers, fiiTickers, usTickers, startDate='2018-01-01', cache='debug/cache_dividends.tsv'):
        self.brTickerList = brTickers
        self.usTickerList = usTickers
        self.fiiList = fiiTickers
        self.startDate = startDate
        self.df = pd.DataFrame(columns=['SYMBOL', 'PRICE', 'PAYDATE', 'OPERATION'])
        self.cache = StockInfoCache(cache)

    def load(self):
        if not self.cache.is_updated():
            if(self.brTickerList != None and len(self.brTickerList) > 0):
                self.df = self.loadData(self.brTickerList, type='ação')
            
            if(self.fiiList != None and  len(self.fiiList) > 0):
                tmp = self.loadData(self.fiiList, 'fii')
                self.df = tmp if self.df.empty else pd.concat([self.df, tmp])

            if(self.usTickerList != None and len(self.usTickerList) > 0):
                tmp = self.loadData(self.usTickerList, 'stock')
                self.df = tmp if self.df.empty else pd.concat([self.df, tmp])

            self.df = self.cache.merge(self.df, sortby=['DATE', 'SYMBOL'], on=['SYMBOL', 'DATE', 'OPERATION'])
        else:      
            self.df = self.cache.load_data()
            self.df['PAYDATE'] = pd.to_datetime(self.df['PAYDATE'], format='%Y/%m/%d')

        if not self.df.empty:
            self.df.set_index('DATE', inplace = True)
            self.df['PRICE'] -= self.df['TAX']
            self.df['OPERATION'] = self.df['OPERATION'].map(lambda x: 'D' if x == 'JCP' else x )
            self.df = self.df[['SYMBOL', 'PRICE', 'PAYDATE', 'OPERATION']]

    def loadData(self, paperList, type):
        tb = pd.DataFrame()
        # pageObj = ADVFN_Page()
        pageObj = Fundamentus_Page(type)

        for paper in paperList:
            rawTable = pageObj.read(paper)
            if(rawTable.empty):
                continue

            # print(rawTable)
            rawTable['SYMBOL'] = paper

            # Discount a tax of 15% when is JCP (Juros sobre capital proprio)
            rawTable['TAX'] = np.where(rawTable['OPERATION'] == 'JCP', rawTable['PRICE'] * 0.15 , 0)
            
            rawTable['PAYDATE'] = np.where(rawTable['PAYDATE'] == '-', rawTable['DATE'], rawTable['PAYDATE'])
            rawTable['PAYDATE'] = pd.to_datetime(rawTable['PAYDATE'], format='%d/%m/%Y')
            rawTable['DATE'] = pd.to_datetime(rawTable['DATE'], format='%d/%m/%Y')
            rawTable = rawTable[['SYMBOL', 'DATE','PRICE', 'PAYDATE', 'OPERATION', 'TAX']]

            tb = pd.concat([tb, rawTable])
        # print(tb)
        return tb[tb['DATE'] >= self.startDate]

    def getPeriod(self, paper, fromDate, toDate):
        filtered = self.df[self.df['SYMBOL'] == paper].loc[fromDate:toDate]        
        return filtered[['SYMBOL', 'PRICE', 'PAYDATE', 'OPERATION']]

#     -------------------------------------------------------------------------------------------------

import yfinance as yf

class YfinanceReader(DividendReader):
    def loadData(self, paperList, type=None):
        res = pd.DataFrame()
        
        for paper in paperList:
            try:
                data = pd.DataFrame(yf.Ticker(paper).dividends)
            except:
                continue

            data['SYMBOL'] = paper.replace('.SA','')
            res = pd.concat([res,data], axis=0)

        res.reset_index(inplace=True)
        res.rename(columns={'Date':'DATE', 'Dividends':'PRICE'}, inplace=True)
        res['PAYDATE'] = res['DATE'] = pd.to_datetime(res['DATE'], format='%Y/%m/%d')
        res = res[res['DATE'] >= self.startDate]
        # 30% tax
        res['TAX'] = res['PRICE'] * 0.3 * 0
        # print(res)
        res = res[['SYMBOL', 'DATE','PRICE', 'PAYDATE', 'TAX']]
        res['OPERATION'] = 'D2'
        return res

#     -------------------------------------------------------------------------------------------------

class SplitsReader:
    def __init__(self, brTickers, usTickers, startDate='2018-01-01', cache='debug/cache_splits.tsv'):
        self.brTickerList = [ t + '.SA' for t in brTickers]
        self.usTickerList = usTickers if usTickers is not None else []
        self.startDate=startDate
        self.df = pd.DataFrame()
        self.cache = StockInfoCache(cache)
    
    def load(self):
        if not self.cache.is_updated():
            if(len(self.brTickerList) > 0):
                self.df = pd.concat([self.df, self.loadData(self.brTickerList)])
            
            if(len(self.usTickerList) > 0):
                self.df = pd.concat([self.df, self.loadData(self.usTickerList)])

            self.df = self.cache.merge(self.df)
        else:      
            self.df = self.cache.load_data()

        self.df.set_index('DATE', inplace = True)

    def getPeriod(self, ticker, fromDate, toDate):
        filtered = self.df[self.df['SYMBOL'] == ticker].loc[fromDate:toDate]
        return filtered[['SYMBOL', 'QUANTITY']]

    def loadData(self, tickerList):
        res = pd.DataFrame()
        for ticker in tickerList:
            try:
                data = pd.DataFrame(yf.Ticker(ticker).splits)
            except:
                continue
            data['SYMBOL'] = ticker.replace('.SA', '')
            res = pd.concat([res,data], axis=0)
        res.index.rename('DATE', inplace=True)
        res.columns = ['QUANTITY', 'SYMBOL']
        res = res.reset_index()
        res['DATE'] = pd.to_datetime(res['DATE'], format='%Y-%m-%d')
        return res[res['DATE'] > self.startDate]

#     -------------------------------------------------------------------------------------------------

class TableAccumulator:
    def __init__(self, pcr = None):
        self.cash = self.avr = self.brl_avr = self.acumQty = self.acumProv=0
        self.pcr = pcr

    def get_currency_rate(self, date):
        currency_rate = 1
        if not self.pcr == None:
            currency_rate = self.pcr.getIndexCurrentValue('USD', date)
        return currency_rate

    def ByRow(self, row):
        total = row.loc['AMOUNT']
        stType = row.loc['OPERATION']
        qty = row.loc['QUANTITY']
        currency_rate = self.get_currency_rate(row['DATE'])

        # buy
        if (stType == 'B'):
            operationValue = row.loc['PRICE'] * qty + row.loc['FEE']
            self.avr = (self.avr * self.acumQty) + operationValue
            self.brl_avr = (self.brl_avr * self.acumQty) + (operationValue / currency_rate)
            self.acumQty += qty
            self.avr /= self.acumQty
            self.brl_avr /= self.acumQty            

        # Sell
        elif (stType == 'S'):
            self.acumQty += qty
            if (self.acumQty == 0):
                self.acumProv = 0

        # Amortization
        elif (stType in ['A']):
            total = np.nan
            row['QUANTITY'] = self.acumQty
            if( self.acumQty > 0 ):
                operationValue = row.loc['PRICE'] * self.acumQty + row.loc['FEE']
                self.avr = ((self.avr * self.acumQty) - operationValue) / self.acumQty
                total = row.loc['PRICE'] * self.acumQty
                self.acumProv += total

        # Split
        elif (stType == "SPLIT"):
            self.acumQty *= qty
            self.avr /= qty

        # Dividend
        elif (stType in ["D", 'R', 'JCP']):
            total = np.nan
            if row['QUANTITY'] == 0 and  self.acumQty != 0:
              # Means the price represents the total
                row['PRICE'] /= self.acumQty

            row['QUANTITY'] = self.acumQty
            if( self.acumQty > 0 ):
                total = row.loc['PRICE'] * row['QUANTITY']
                self.acumProv += total

        # Dividend, Tax, Amortization
        elif (stType in ["D1", 'R1', 'JCP1', 'T1', 'A1', 'I1']):
            total = row.loc['PRICE'] * row['QUANTITY']
            if stType != 'I1':
              self.acumProv += total

        row['AMOUNT'] = total
        row['acumProv'] = self.acumProv
        row['acum_qty'] = self.acumQty
        row['PM'] = self.avr
        row['PM_BRL'] = self.brl_avr
        return row

    def ByGroup(self, group):
        self.avr = self.brl_avr = self.acumQty = self.acumProv = 0
        return group.apply(self.ByRow, axis=1)

    def Cash(self, row):
        stType = row.loc['OPERATION']
        amount = round(row.loc['AMOUNT'], 6)

        if (stType in ['C', 'W']):
            self.cash += amount + row.loc['FEE']
            row.loc['acum_qty'] = row.loc['QUANTITY']
            row.loc['PM'] = row.loc['PRICE']
            row['PM_BRL'] = row.loc['PRICE'] / self.get_currency_rate(row['DATE'])

        elif (stType in ['B', 'S']):
            self.cash -= (amount + row.loc['FEE'])
        
        elif ((stType in ['D1', 'A1', 'R1', 'JCP1', 'T1', 'I1', 'CF']) or (stType in ['D', 'A', 'R', 'JCP', 'T'] and row['acum_qty'] > 0)):
            # self.acumProv += amount 
            self.cash += amount

        row['CASH'] = round(self.cash, 6)
        return row
#     -------------------------------------------------------------------------------------------------

#Class to calculate the profit or loss considering day trade rules.
class Profit:
    def __init__(self):
        self.pm = self.amount = 0
    
    def DayTrade(self, row):
        profit = 0
        amount = self.amount + row.QUANTITY
        if(row.OPERATION == "B"):
            self.pm = (row.PRICE * row.QUANTITY) / amount
        else:
            profit = (self.pm - row.PRICE) * row.QUANTITY
            amount = self.amount - row.QUANTITY

        self.amount = amount
        row['Profit'] = profit
        row['DayTrade'] = 1
        return row
        
    def Trade(self, dayGroup):
        purchaseDf = dayGroup.loc[dayGroup.OPERATION == 'B']
        sellDf = dayGroup.loc[dayGroup.OPERATION == 'S']
        
        sellCount = len(sellDf)
        purchaseCount = len(purchaseDf)
        
        if(sellCount == 0):
            dayGroup['Profit'] = dayGroup['DayTrade'] = 0
            return dayGroup
         
        if(purchaseCount == 0):
            dayGroup['Profit'] = ((dayGroup.PRICE - dayGroup.PM) * -dayGroup.QUANTITY ) - dayGroup.FEE
            dayGroup['DayTrade'] = 0
            return dayGroup

        # Day trade detected
        # print('Day Trade detected\n', dayGroup)
        self.pm = self.amount = 0
        return dayGroup.apply(self.DayTrade, axis=1)

#     -------------------------------------------------------------------------------------------------
class Portifolio:
    def __init__(self, priceReader, dFrame, recommended=None, currency='$'):
        self.currency = currency
        self.dtframe = dFrame.groupby(['SYMBOL']).apply(lambda x: x.tail(1))

        dFrame = dFrame.sort_values(['PAYDATE', 'OPERATION'], ascending=[True, False])
        dFrame=dFrame.apply(TableAccumulator().Cash, axis=1)
        cash = dFrame.iloc[-1]['CASH']

        self.dtframe = self.dtframe[['SYMBOL', 'PM', 'acum_qty', 'acumProv', 'TYPE']]
        self.dtframe.columns = ['SYMBOL', 'PM', 'QUANTITY', 'DIVIDENDS', 'TYPE']
        self.dtframe["COST"] = self.dtframe.PM * self.dtframe['QUANTITY']
        self.dtframe = self.dtframe[self.dtframe['QUANTITY'] > 0]
        self.dtframe.reset_index(drop=True, inplace=True)

        self.dtframe =  self.dtframe[ self.dtframe['SYMBOL'] != 'CASH']
        self.dtframe['PRICE'] = self.dtframe.apply(priceReader.fillCurrentValue, axis=1)['PRICE']
        self.dtframe['PRICE'] = self.dtframe['PRICE'].fillna(self.dtframe['PM'])
        self.dtframe["MKT_VALUE"] = self.dtframe['PRICE'] * self.dtframe['QUANTITY']
        
        newLine = {'SYMBOL':'CASH', 'PM':cash, 'QUANTITY':1, 'DIVIDENDS':0, 'TYPE':'C', 'COST':cash, 'PRICE':cash, 'MKT_VALUE':cash}
        self.dtframe = pd.concat([self.dtframe, pd.DataFrame(newLine, index=[0])])
        
        self.dtframe[f'GAIN({currency})'] = self.dtframe['MKT_VALUE'] - self.dtframe['COST']
        self.dtframe[f'GAIN+DIV({currency})'] = self.dtframe[f'GAIN({currency})'] + self.dtframe['DIVIDENDS']
        self.dtframe['GAIN(%)'] = self.dtframe[f'GAIN({currency})'] / self.dtframe['COST'] *100
        self.dtframe['GAIN+DIV(%)'] = self.dtframe[f'GAIN+DIV({currency})'] / self.dtframe['COST'] * 100
        self.dtframe['ALLOCATION'] = (self.dtframe['MKT_VALUE'] / self.dtframe['MKT_VALUE'].sum()) * 100
        self.dtframe = self.dtframe.replace([np.inf, -np.inf], np.nan).fillna(0)

        self.dtframe = self.dtframe[self.dtframe['PM'] > 0]

        self.dtframe = self.dtframe[['SYMBOL', 'PM', 'PRICE', 'QUANTITY', 'COST', 'MKT_VALUE', 'DIVIDENDS', f'GAIN({currency})', f'GAIN+DIV({currency})',\
                                     'GAIN(%)', 'GAIN+DIV(%)', 'ALLOCATION']]

        self.format = {'PRICE': f'{currency} {{:,.2f}}', 'PM': f'{currency} {{:,.2f}}', 'QUANTITY': '{:>n}', 'COST': f'{currency} {{:,.2f}}',\
                       'MKT_VALUE': f'{currency} {{:,.2f}}', 'DIVIDENDS': f'{currency} {{:,.2f}}', f'GAIN({currency})': f'{currency} {{:,.2f}}',\
                    f'GAIN+DIV({currency})': f'{currency} {{:,.2f}}', 'GAIN(%)': '{:,.2f}%', 'GAIN+DIV(%)': '{:,.2f}%', 'ALLOCATION': '{:,.2f}%'}

        self.extra_content(recommended)

        self.dtframe.set_index('SYMBOL', inplace=True)

    def extra_content(self, recommended):
        if recommended == None:
            return

        self.dtframe['TARGET'], self.dtframe['TOP_PRICE'], self.dtframe['PRIORITY'] = zip(*self.dtframe['SYMBOL'].map(lambda x: self.recommended(recommended, x)))
        self.dtframe['BUY'] = (self.dtframe['QUANTITY'] * (self.dtframe['TARGET']/100 - self.dtframe['ALLOCATION']/100)) / (self.dtframe['ALLOCATION']/100)
        format = {'TARGET': '{:,.2f}%', 'TOP_PRICE': f'{self.currency} {{:,.2f}}', 'BUY': '{:,.1f}'}
        self.format = {**self.format , **format}
    
    def recommended(self, recom, symbol):
        for ticker in recom['Tickers']:
            if symbol == ticker['Ticker']:
                return float(ticker['Participation']) * 100, float(ticker['Top']), int(ticker['Priority'])
        return 0, 0, 99

    def show(self):
        fdf = self.dtframe
        return fdf.style.applymap(color_negative_red)\
               .format(self.format)

#     -------------------------------------------------------------------------------------------------
class PerformanceBlueprint:
    def __init__(self, priceReader, dataframe, date, currency='R$'):
        self.currency=currency
        self.pcRdr = priceReader
        self.equity = self.cost = self.realizedProfit = self.div = self.paperProfit = self.profit \
        = self.usdIbov = self.ibov = self.sp500 = self.profitRate = self.expense = 0
        self.date = date
        self.df = dataframe[(dataframe['DATE'] <= date)].copy(deep=True)
        if (not self.df.empty):
            priceReader.setFillDate(self.date)
            self.pt = Portifolio(self.pcRdr,self.df)

    def calc(self):
        if (not self.df.empty):
            ptf = self.pt.dtframe
            self.equity          = (ptf['PRICE'] * ptf['QUANTITY']).sum()
            self.cost            = ptf['COST'].sum()
            self.realizedProfit  = self.df.loc[self.df.OPERATION == 'S', 'Profit'].sum()
            self.div             = self.df[self.df.OPERATION.isin(['D1', 'A1', 'R1', 'JCP1', 'D', 'A', 'R', 'JCP', 'CF'])]['AMOUNT'].sum()
            self.paperProfit     = self.equity -    self.cost
            self.profit          = self.equity -    self.cost +    self.realizedProfit +    self.div
            self.profitRate      = self.profit / self.cost
            indexHistory         = self.pcRdr.getIndexHistory('IBOV',self.date)
            self.ibov            = indexHistory.iloc[-1]/indexHistory.iloc[0] - 1
            indexHistory         = self.pcRdr.getIndexHistory('S&P500', self.date)
            self.sp500           = indexHistory.iloc[-1]/indexHistory.iloc[0] - 1
            self.selic           = self.pcRdr.getIndexCurrentValue('selic', self.date)
            self.expense         = self.df.loc[self.df.OPERATION == "B",'FEE'].sum()
            self.exchangeRatio   = self.pcRdr.getIndexCurrentValue('USD', self.date)
            return self

#     -------------------------------------------------------------------------------------------------

import matplotlib.pyplot as plt
#Class to calculate the average price by Stock group
class Acumulator:
    def __init__(self):
        self.acumulated=0 

    def calcLoss(self, row):
        acumulated = self.acumulated

        if(row.loc['Profit'] < 0 or acumulated < 0):
            acumulated = acumulated + row.loc['Profit']
        
        if(acumulated > 0):
            acumulated = 0

        self.acumulated = acumulated
        return self.acumulated
class PerformanceViewer:
    def __init__(self, *args):
        self.pf = pd.DataFrame(columns = ['Item', 'BRL', 'USD', '%'])
        if (len(args) == 2 and isinstance(args[0], pd.DataFrame)):
            row = args[0].set_index('Date').loc[args[1]]
            self.buildTable(row['Equity'], row['Cost'], row['Expense'], row['paperProfit'], row['Profit'], row['Div'], row['TotalProfit'])
        elif(isinstance(args[0], PerformanceBlueprint)):
            p = args[0]
            self.buildTable(p.equity, p.cost, p.expense, p.paperProfit, p.realizedProfit, p.div, p.profit, p.currency, p.exchangeRatio)

    def buildTable(self, equity, cost, expense, paperProfit, profit, div, totalProfit, currency='$', exchangeRatio=0.22):
        self.pf.loc[len(self.pf)] = ['Equity          ' , equity,equity, equity/cost]
        self.pf.loc[len(self.pf)] = ['Cost            ' , cost,cost, 1]
        self.pf.loc[len(self.pf)] = ['Expenses        ' , expense,expense, expense/cost]
        self.pf.loc[len(self.pf)] = ['Paper profit    ' , paperProfit,paperProfit, paperProfit/cost]
        self.pf.loc[len(self.pf)] = ['Realized profit ' , profit,profit, profit/cost]
        self.pf.loc[len(self.pf)] = ['Dividends       ' , div,div, div/cost]
        self.pf.loc[len(self.pf)] = ['Total Profit    ' , totalProfit,totalProfit, totalProfit/cost]
        self.pf.loc[:, '%'] *= 100
        if currency == '$':
            self.pf.loc[:, 'BRL'] /= exchangeRatio
        else:
            self.pf.loc[:, 'USD'] *= exchangeRatio
        self.pf.set_index('Item', inplace=True)

    def show(self):
        format_dict = { 'USD': ' {:^,.2f}', 'BRL': ' {:^,.2f}', '%': ' {:>.1f}%' }
        return self.pf.style.applymap(color_negative_red).format(format_dict)

#     -------------------------------------------------------------------------------------------------

from math import ceil
import string
import re

class CompanyListReader:
        def __init__(self):
                self.dtFrame = self.loadBmfBovespa()
        
        def loadInfomoney(self):
                url = 'https://www.infomoney.com.br/minhas-financas/confira-o-cnpj-das-acoes-negociadas-em-bolsa-e-saiba-como-declarar-no-imposto-de-renda/'
                r = requests.get(url, headers=http_header)
                rawTable = pd.read_html(r.text, thousands='.',decimal=',')[0]
                return rawTable

        def loadBmfBovespa(self):
                url = 'https://bvmf.bmfbovespa.com.br/CapitalSocial/'
                try:
                    r = requests.get(url, headers=http_header, timeout=5)
                except:
                    print(f'Error to read url: {url}')
                    return pd.DataFrame()

                rawTable = pd.read_html(r.text, thousands='.',decimal=',')[0]
                rawTable = rawTable.iloc[:, :4] # Remove columns after 4th column.
                rawTable.columns = ['NAME', 'CODE', 'SOCIAL_NAME', 'SEGMENT']
                return rawTable

        def loadOceans(self):
                url = 'https://www.oceans14.com.br/acoes/'
                r = requests.get(url, headers=http_header)
                rawTable = pd.read_html(r.text, thousands='.',decimal=',')[0]
                return rawTable

        def loadGuiaInvest(self):
                pageAmount = 10
                rawTable = pd.DataFrame()
                url = 'https://www.guiainvest.com.br/lista-acoes/default.aspx?listaacaopage='
                r = requests.get(url, headers=http_header)
                df = pd.read_html(r.text, thousands='.',decimal=',')[0]
                res = re.search('Registros\s\d+\s-\s(\d+)\sde\s(\d+)', df.to_string())
                if (res):
                        pageAmount = ceil(int(res.group(2))/ int(res.group(1)))

                for i in range(1, pageAmount):
                    r = requests.get(url + str(i), headers=http_header)
                    rawTable = pd.concat([rawTable, pd.read_html(r.text, thousands='.',decimal=',')[0].drop(['Unnamed: 0', 'Atividade Principal'], axis=1)])

                return rawTable.reset_index(drop=True)

        def loadAdvfn(self):
                rawTable = pd.DataFrame()
                url = 'https://br.advfn.com/bolsa-de-valores/bovespa/'
                for pg in string.ascii_uppercase:    
                    r = requests.get(url + pg, headers=http_header)
                    # rawTable = rawTable.append(pd.read_html(r.text, thousands='.',decimal=',')[0])
                    rawTable = pd.concat([rawTable, pd.read_html(r.text, thousands='.',decimal=',')[0]])

                return rawTable[['Ação',	'Unnamed: 1']].dropna()

        def loadFundamentus(self):
                def SubCategory(row):
                        if ('3' in row['Paper']):
                                row['Sub'] = 'ON'
                        elif ('4' in row['Paper']):
                                row['Sub'] = 'PN'
                        else:
                                row['Sub'] = 'UNT'
                        return row

                url = 'https://www.fundamentus.com.br/detalhes.php?papel='
                r = requests.get(url, headers=http_header)
                rawTable = pd.read_html(r.text, thousands='.',decimal=',')[0].fillna('Unown')
                rawTable.columns = ['Paper', 'Company', 'FullName']
                rawTable = rawTable.apply(lambda x: x.str.upper())

                rawTable = rawTable.apply(SubCategory,axis=1)        
                return rawTable
#     -------------------------------------------------------------------------------------------------

##Clear operation cost before 2019
def clear2018Cost(row):
    if (row['DATE'].year < 2019):
        return 0
    return row['FEE']

def color_negative_red(val):
    return 'color: %s' % ('red' if val < 0 else 'green')

if __name__ == "__main__":
    tickers = ['ABEV3', 'BBDC3', 'BMEB4', 'CARD3', 'CIEL3', 'COGN3', 'ECOR3', 'EGIE3', 'EZTC3', 'FLRY3', 'GOLL4', 'GRND3', 'HGTX3', 'ITUB3', 'KLBN11', 'LCAM3', 'MDIA3', 'MOVI3', 'MRVE3', 'OIBR3', 'PARD3', 'PETR4', 'PRIO3', 'PSSA3', 'SBFG3', 'SMLS3', 'TASA4', 'TRIS3', 'VVAR3', 'WEGE3', 'XPBR31', 'BBFI11B', 'DEVA11', 'FAMB11B', 'FIGS11', 'GTWR11', 'HGRE11', 'HGRU11', 'HSLG11', 'HSML11', 'HTMX11', 'KNSC11', 'MFII11', 'MXRF11', 'RBRF11', 'RBRY11', 'RVBI11', 'SPTW11', 'VILG11', 'VISC11', 'VRTA11', 'XPCM11', 'XPLG11', 'XPML11']
    tickers_us = ['CSCO', 'VZ', 'LUMN', 'EA', 'NEM', 'KWEB', 'PRIM', 'HOLI']
    # prcReader = PriceReader(tickers,[])
    # prcReader.load()
    # print(prcReader.df)
    # print(prcReader.brlIndex)
    # print(prcReader.getCurrentValue('CCJ', '2018-02-14'))

    DividendReader(tickers, None, None).load()
    # YfinanceReader(tickers_us, None, None).load()

    
    # dr = SplitsReader(['ABEV3', 'BBDC3', 'BMEB4', 'CARD3', 'CIEL3', 'COGN3', 'ECOR3', 'EGIE3', 'EZTC3', 'FLRY3', 'GOLL4', 'GRND3', 'HGTX3', 'ITUB3', 'KLBN11', 'LCAM3', 'MDIA3', 'MOVI3', 'MRVE3', 'OIBR3', 'PARD3', 'PETR4', 'PRIO3', 'PSSA3', 'SBFG3', 'SMLS3', 'TASA4', 'TRIS3', 'VVAR3', 'WEGE3', 'XPBR31'], [], '2018-03-14 00:00:00')
    # dr.load()
    # print(dr.df)
