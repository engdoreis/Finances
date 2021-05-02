from pandas_datareader import data as web
import pandas as pd
import datetime as dt
import numpy as np

class PriceReader:
    def __init__(self, brTickerList, usTickerList, startDate='01-01-2018'):
        self.brTickerList = brTickerList
        self.usTickerList = usTickerList
        self.startDate = startDate
        self.fillDate = dt.datetime.today().strftime('%m-%d-%Y')
        self.df = pd.DataFrame(columns=['Date'])

    def load(self):
        # Read BR market data
        if((self.brTickerList != None) and (len(self.brTickerList) > 0)):
            self.df = self.readData(self.brTickerList, self.startDate).reset_index()
            self.df.columns = self.df.columns.str.replace('\.SA','')

        # Read US Market data
        if((self.usTickerList != None) and (len(self.usTickerList) > 0)):
            self.df = self.df.merge(self.readUSData(self.usTickerList, self.startDate).reset_index(), how='outer', on='Date')

        self.df = self.df.set_index('Date').sort_index()
        self.df.to_csv('debug.csv', sep='\t')

        indexList = ['^BVSP', '^GSPC', 'BRLUSD=X']
        self.brlIndex = self.readUSData(indexList, self.startDate).reset_index()
        self.brlIndex.columns = ['Date', 'IBOV', 'S&P500', 'USD']
        self.brlIndex = self.brlIndex.set_index('Date')
        # display(self.brlIndex)

    def setFillDate(self, date):
        self.fillDate = date

    def fillCurrentValue(self, row):
        row['PRICE'] = self.getCurrentValue(row['SYMBOL'], self.fillDate)
        return row

    def readData(self, code, startDate='01-01-2018'):
        s=[]
        for c in code:
            s.append(c + '.SA')
        # importar dados para o DataFrame
        return web.DataReader(s, data_source='yahoo', start=startDate)['Close']

    def readUSData(self, code, startDate='01-01-2018'):
        # importar dados para o DataFrame
        return web.DataReader(code, data_source='yahoo', start=startDate)['Close']

    def getHistory(self, code, start='01-01-2018'):
        return self.df.loc[start:][code]

    def getCurrentValue(self, code, date=None):
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

import requests
class DividendReader:
    fiiUrl = 'https://www.fundamentus.com.br/fii_proventos.php?papel={}&tipo=2'
    stockUrl = 'https://www.fundamentus.com.br/proventos.php?papel={}&tipo=2'
    notFound = 'Nenhum provento encontrado'
    header = {
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.75 Safari/537.36",
            "X-Requested-With": "XMLHttpRequest"
        }

    def __init__(self, dataFrame):
        self.brTickerList = dataFrame[dataFrame['TYPE'] == 'Ação']['SYMBOL'].unique()
        self.usTickerList = dataFrame[dataFrame['TYPE'] == 'STOCK']['SYMBOL'].unique()
        self.fiiList = dataFrame[dataFrame['TYPE'] == 'FII']['SYMBOL'].unique()
        self.df = pd.DataFrame(columns=['SYMBOL', 'PRICE', 'Data de Pagamento'])

    def load(self):
        if(len(self.brTickerList) > 0):
            self.df = self.df.append(self.loadData(self.brTickerList, self.stockUrl))
        
        if(len(self.fiiList) > 0):
            self.df = self.df.append(self.loadData(self.fiiList, self.fiiUrl))

        if(not self.df.empty):
            self.df = self.df.sort_values(by=['DATE', 'SYMBOL'])
            self.df.set_index("Data", inplace = True)
            self.df = self.df[['SYMBOL', 'PRICE', 'Data de Pagamento']]
            # display(self.df.tail(20))

    def loadData(self, paperList, baseUrl):
        tb = pd.DataFrame()
        for paper in paperList:
            url = baseUrl.format(paper)
            # print(f'\n\nSearching: {url}')
            r = requests.get(url, headers=self.header)
            if(self.notFound in r.text):
                # print(self.notFound )
                continue

            rawTable = pd.read_html(r.text, thousands='.',decimal=',')[0]
            if('fii' in baseUrl):
                rawTable.columns = ['DATE', 'OPERATION', 'Data de Pagamento', 'PRICE'] 

            rawTable['SYMBOL'] = paper
            if('Por quantas ações' in rawTable.columns):
                rawTable['PRICE'] /= rawTable['Por quantas ações']
                # Discount a taxe of 15% when is JCP (Juros sobre capital proprio)
                rawTable['PRICE'] = np.where(rawTable['OPERATION'] == 'DIVIDENDO',    rawTable['PRICE'],    rawTable['PRICE'] * 0.85 )
                

            rawTable = rawTable[['SYMBOL', 'DATE','PRICE', 'Data de Pagamento']]

            rawTable['Data de Pagamento'] = np.where(rawTable['Data de Pagamento'] == '-',\
                            rawTable['DATE'], rawTable['Data de Pagamento'])
            
            rawTable['Data de Pagamento'] = pd.to_datetime(rawTable['Data de Pagamento'], format='%d/%m/%Y')
            rawTable['DATE'] = pd.to_datetime(rawTable['DATE'], format='%d/%m/%Y')

            # display(rawTable.tail())
            tb = tb.append(rawTable)
        return tb

    def getPeriod(self, paper, fromDate, toDate):
        filtered = self.df[self.df['SYMBOL'] == paper].loc[fromDate:toDate]
        return filtered[['SYMBOL', 'PRICE']]
    # display(tb)

#     -------------------------------------------------------------------------------------------------

import yfinance as yf

class YfinanceReader(DividendReader):
    def loadData(self, paperList, baseUrl):
        res = pd.DataFrame()
        for paper in paperList:
            data = pd.DataFrame(yf.Ticker(paper + '.SA').dividends)
            data['SYMBOL'] = paper
            res = pd.concat([res,data], axis=0)
        res.index.rename('DATE', inplace=True)
        res.columns = ['PRICE', 'SYMBOL']
        res['Data de Pagamento'] = 0
        # display(res[res['SYMBOL'] == 'CIEL3'])
        return res[['SYMBOL', 'PRICE', 'Data de Pagamento']].reset_index()

#     -------------------------------------------------------------------------------------------------

class SplitsReader:
    def __init__(self, dataFrame):
        self.brTickerList = dataFrame[dataFrame['TYPE'].isin(['Ação'])]['SYMBOL'].unique()
        self.brTickerList += '.SA'
        self.usTickerList = dataFrame[dataFrame['TYPE'].isin(['STOCK'])]['SYMBOL'].unique()
        self.df = pd.DataFrame()
    
    def load(self):
        if(len(self.brTickerList) > 0):
            self.df = self.df.append(self.loadData(self.brTickerList))
        
        if(len(self.usTickerList) > 0):
            self.df = self.df.append(self.loadData(self.usTickerList))

        self.df = self.df.sort_values(by=['DATE', 'SYMBOL'])
        self.df.set_index('DATE', inplace = True)

    def getPeriod(self, ticker, fromDate, toDate):
        filtered = self.df[self.df['SYMBOL'] == ticker].loc[fromDate:toDate]
        return filtered[['SYMBOL', 'QUANTITY']]

    def loadData(self, tickerList):
        res = pd.DataFrame()
        for ticker in tickerList:
            # print(ticker)
            try:
                data = pd.DataFrame(yf.Ticker(ticker).splits)
            except:
                continue
            data['SYMBOL'] = ticker.replace('.SA', '')
            res = pd.concat([res,data], axis=0)
        res.index.rename('DATE', inplace=True)
        res.columns = ['QUANTITY', 'SYMBOL']
        # display(res)
        return res.reset_index()

#     -------------------------------------------------------------------------------------------------

class TableAccumulator:
    def __init__(self):
        self.cash = self.avr = self.acumQty = self.acumProv=0

    def ByRow(self, row):
        total = row.loc['AMOUNT']
        stType = row.loc['OPERATION']
        qty = row.loc['QUANTITY']

        if (stType == 'B'):
            operationValue = row.loc['PRICE'] * qty + row.loc['FEE']
            self.avr = ((self.avr * self.acumQty) + operationValue) 
            self.acumQty += qty
            self.avr /= self.acumQty

        elif (stType == 'S'):
            self.acumQty += qty
            if (self.acumQty == 0):
                self.acumProv = 0

        elif (stType == "SPLIT"):
            self.acumQty *= qty
            self.avr /= qty

        elif (stType == "DIVIDENDS"):
            total = np.nan
            row['QUANTITY'] = self.acumQty
            if( self.acumQty > 0 ):
                total = row.loc['PRICE'] * self.acumQty
                self.acumProv += total
 

        row['AMOUNT'] = total
        row['acumProv'] = self.acumProv
        row['acum_qty'] = self.acumQty
        row['PM'] = self.avr
        return row

    def ByGroup(self, group):
        self.avr = self.acumQty = self.acumProv = 0
        return group.apply(self.ByRow, axis=1)

    def Cash(self, row):
        stType = row.loc['OPERATION']
        amount = row.loc['QUANTITY'] * row.loc['PRICE']

        if (stType in ['S', 'W']):
            self.cash += amount + row.loc['FEE']

        elif (stType == 'B'):
            self.cash -= amount - row.loc['FEE']
        
        elif (stType == "DIVIDENDS" and row['acum_qty'] > 0):
            self.acumProv += amount 

        row['CASH'] = self.cash
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
    def __init__(self, priceReader, dFrame):
        self.dtframe = dFrame.groupby(['SYMBOL']).apply(lambda x: x.tail(1) )
        cash = dFrame.iloc[-1]['CASH']
        self.dtframe = self.dtframe[['SYMBOL', 'PM', 'acum_qty', 'acumProv', 'TYPE']]
        self.dtframe.columns = ['SYMBOL', 'PM', 'QUANTITY', 'DIVIDENDS', 'TYPE']
        self.dtframe.reset_index(drop=True, inplace=True)
        self.dtframe["COST"] = self.dtframe.PM * self.dtframe.QUANTITY
        self.dtframe = self.dtframe[self.dtframe['QUANTITY'] > 0]

        self.dtframe['PRICE'] = self.dtframe.apply(priceReader.fillCurrentValue, axis=1)['PRICE']
        self.dtframe['PRICE'] = self.dtframe['PRICE'].fillna(self.dtframe['PM'])
        self.dtframe["MKT_VALUE"] = self.dtframe['PRICE'] * self.dtframe.QUANTITY
        
        newLine = {'SYMBOL':'CASH', 'PM':cash, 'QUANTITY':1, 'DIVIDENDS':0, 'TYPE':'W', 'COST':cash, 'PRICE':cash, 'MKT_VALUE':cash}
        self.dtframe = self.dtframe.append(pd.DataFrame(newLine, index=[0]))
        # self.dtframe.to_csv('H:/Git/Finances/log1.csv')
        
        self.dtframe['GAIN($)'] = self.dtframe['MKT_VALUE'] - self.dtframe['COST']
        self.dtframe['GAIN+DIV($)'] = self.dtframe['GAIN($)'] + self.dtframe['DIVIDENDS']
        self.dtframe['GAIN(%)'] = self.dtframe['GAIN($)'] / self.dtframe['COST'] *100
        self.dtframe['GAIN+DIV(%)'] = self.dtframe['GAIN+DIV($)'] / self.dtframe['COST'] * 100
        self.dtframe['ALLOCATION'] = (self.dtframe['MKT_VALUE'] / self.dtframe['MKT_VALUE'].sum()) * 100
        self.dtframe = self.dtframe.replace([np.inf, -np.inf], np.nan).fillna(0)

        self.dtframe = self.dtframe[['SYMBOL', 'PM', 'PRICE', 'QUANTITY', 'COST', 'MKT_VALUE', 'DIVIDENDS', 'GAIN($)', 'GAIN+DIV($)', 'GAIN(%)', 'GAIN+DIV(%)', 'ALLOCATION']]
        self.dtframe.set_index('SYMBOL', inplace=True)
        self.format = {'PRICE': '$ {:,.2f}', 'PM': '$ {:.2f}', 'QUANTITY': '{:>n}', 'COST': '$ {:,.2f}', 'MKT_VALUE': '$ {:,.2f}', 'DIVIDENDS': '$ {:,.2f}',\
                                    'GAIN($)': '$ {:,.2f}', 'GAIN+DIV($)': '$ {:,.2f}', 'GAIN(%)': '{:,.2f}%', 'GAIN+DIV(%)': '{:,.2f}%', 'ALLOCATION': '{:,.2f}%'}

    def show(self):
        fdf = self.dtframe
        # fdf.loc['AMOUNT', 'COST'] = fdf['COST'].sum()
        # fdf.loc['AMOUNT', 'MKT_VALUE'] = fdf['MKT_VALUE'].sum()
        # fdf.loc['AMOUNT', 'GAIN($)'] = fdf['GAIN($)'].sum()
        # fdf.fillna(' ', inplace=True)
        return fdf.style.applymap(color_negative_red).format(self.format)

#     -------------------------------------------------------------------------------------------------
class PerformanceBlueprint:
    def __init__(self, priceReader, dataframe, date):
        self.pcRdr = priceReader
        self.equity = self.cost = self.realizedProfit = self.div = self.paperProfit = self.profit \
        = self.usdIbov = self.ibov = self.sp500 = self.profitRate = self.expense = 0
        self.date = date
        self.df = dataframe[(dataframe['DATE'] <= date)]
        # display(self.df)
        if (not self.df.empty):
            priceReader.setFillDate(self.date)
            self.pt = Portifolio(self.pcRdr,self.df)

    def calc(self):
        if (not self.df.empty):
            ptf = self.pt.dtframe
            self.equity          = (ptf['PRICE'] * ptf['QUANTITY']).sum()
            self.cost            = ptf['COST'].sum()
            self.realizedProfit  = self.df.loc[self.df.OPERATION == 'S', 'Profit'].sum()
            self.div             = self.df[self.df.OPERATION == 'DIVIDENDS']['AMOUNT'].sum()
            self.paperProfit     = self.equity -    self.cost
            self.profit          = self.equity -    self.cost +    self.realizedProfit +    self.div
            self.profitRate      = self.profit / self.cost
            indexHistory         = self.pcRdr.getIndexHistory('IBOV',self.date)
            self.ibov            = indexHistory.iloc[-1]/indexHistory.iloc[0] - 1
            indexHistory         = self.pcRdr.getIndexHistory('S&P500', self.date)
            self.sp500           = indexHistory.iloc[-1]/indexHistory.iloc[0] - 1
            self.expense         = self.df.loc[self.df.OPERATION == "B",'FEE'].sum()
            self.exchangeRatio   = self.pcRdr.getIndexCurrentValue('USD',self.date)
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

class Taxation:
    def __init__(self, dataframe, stockTaxFreeMonth=20000, stockTaxRate=0.2, fiiTaxRate=0.2, daytradeTaxRate=0.2):
        self.df = dataframe
        self.stockTaxRate=stockTaxRate
        self.fiiTaxRate=fiiTaxRate
        self.daytradeTaxRate=daytradeTaxRate
        self.stockTaxFreeMonth=stockTaxFreeMonth
        
    def calcStockTaxes(self, dataframe):
        tax = np.where(dataframe['Profit'] > self.stockTaxFreeMonth , dataframe['Profit'] * self.stockTaxRate, 0)
        dataframe['Tax'] = np.where(tax > 0 , tax, 0)

    def calcFiiTaxes(self, dataframe):
        tax = dataframe['Dutiable'] * self.fiiTaxRate
        dataframe['Tax'] = np.where(tax > 0 , tax, 0)

    def calcDaytradeTaxes(self, dataframe):
        tax = dataframe['Dutiable'] * self.daytradeTaxRate
        dataframe['Tax'] = np.where(tax > 0 , tax, 0)

    def DayTrade(self, stockType):
        #Filter by stockType and get the year list
        dayTrade = self.df[(self.df['DayTrade'] == 1) & (self.df['TYPE'] == stockType)]
        dayTrade = dayTrade.groupby(['SYMBOL', 'DATE'])['Profit'].sum().reset_index()
        dayTrade['Year'] = pd.DatetimeIndex(dayTrade['DATE']).year
        dayTrade['Month'] = pd.DatetimeIndex(dayTrade['DATE']).month_name()
        dayTrade=dayTrade[['Month','Profit','Year']]
        return dayTrade

    def SwingTrade(self, stockType):
        swingTrade = pd.DataFrame(columns=['Month','Profit'])
        #Filter by stockType and get the year list
        typeDF = self.df[(self.df['DayTrade'] == 0) & (self.df['TYPE'] == stockType)]
        years = typeDF.Year.unique()
        for year in years: 
            #Calculate the Profit/Loss by month in the current year
            res=typeDF[typeDF.Year == year].groupby(['Month'])['Profit'].sum().reset_index()
            #Sort the table by the month name
            res['Year']=year
            res['m'] = pd.to_datetime(res.Month, format='%B').dt.month
            res.set_index('m', inplace=True)
            res.sort_index(inplace=True)
            res.sort_index(inplace=True)
            res.reset_index(drop=True, inplace=True)
            swingTrade=pd.concat([swingTrade, res], axis=0)

        swingTrade['Year'] = swingTrade['Year'].astype(int)
        return swingTrade

    def Process(self, stockType='FII'):

        if(not self.df['OPERATION'].str.contains(stockType).any()):
            return

        taxDF = self.SwingTrade(stockType)
        if(len(taxDF) > 0): 
            # print('Swingtrade')
            self.swingTradeTable = self.CalcTaxes(taxDF, stockType)

            taxdayTradeDF = self.DayTrade(stockType)
            if(len(taxdayTradeDF) > 0): 
                # print('Daytrade')
                self.dayTradeTable = self.CalcTaxes(taxdayTradeDF, stockType, True)


    def CalcTaxes(self, newDF, stockType, isDaytrade=False):
        # display(newDF)
        acm = Acumulator()
        acumLoss = newDF.apply(acm.calcLoss, axis=1).reset_index()
        # display(acumLoss)
        acumLoss.columns = [ 'Index','AcumLoss']
        acumLoss.set_index('Index', inplace=True)
        newDF=pd.concat([newDF, acumLoss['AcumLoss']], axis=1)

        dutiable = newDF['Profit'] + newDF['AcumLoss'].shift(1, fill_value=0)
        newDF['Dutiable'] = np.where(dutiable > 0, dutiable, 0)
        if (isDaytrade):
            self.calcDaytradeTaxes(newDF)
        elif (stockType == 'FII'):
            self.calcFiiTaxes(newDF)
        else:
            self.calcStockTaxes(newDF)

        newDF.set_index(["Year", 'Month'], inplace=True)
        # display(newDF)
        return(newDF)
        # print( '\n')

#     -------------------------------------------------------------------------------------------------

class PerformanceViewer:
        def __init__(self, *args):
                self.pf = pd.DataFrame(columns = ['Item', 'USD', 'BRL', '%'])
                if (len(args) == 2 and isinstance(args[0], pd.DataFrame)):
                        row = args[0].set_index('Date').loc[args[1]]
                        self.buildTable(row['Equity'], row['Cost'], row['Expense'], row['paperProfit'], row['Profit'], row['Div'], row['TotalProfit'])
                elif(isinstance(args[0], PerformanceBlueprint)):
                        p = args[0]
                        self.buildTable(p.equity, p.cost, p.expense, p.paperProfit, p.realizedProfit, p.div, p.profit, p.exchangeRatio)

        def buildTable(self, equity, cost, expense, paperProfit, profit, div, totalProfit, exchangeRatio=0.22):
                self.pf.loc[len(self.pf)] = ['Equity          ' , equity,equity, equity/cost]
                self.pf.loc[len(self.pf)] = ['Cost            ' , cost,cost, 1]
                self.pf.loc[len(self.pf)] = ['Expenses        ' , expense,expense, expense/cost]
                self.pf.loc[len(self.pf)] = ['Paper profit    ' , paperProfit,paperProfit, paperProfit/cost]
                self.pf.loc[len(self.pf)] = ['Realized profit ' , profit,profit, profit/cost]
                self.pf.loc[len(self.pf)] = ['Dividends       ' , div,div, div/cost]
                self.pf.loc[len(self.pf)] = ['Total Profit    ' , totalProfit,totalProfit, totalProfit/cost]
                self.pf.loc[:, '%'] *= 100
                self.pf.loc[:, 'BRL'] /= exchangeRatio
                self.pf.set_index('Item', inplace=True)

        def show(self):
                format_dict = { 'USD': ' {:^,.2f}', 'BRL': ' {:^,.2f}', '%': ' {:>.1f}%' }
                return self.pf.style.applymap(color_negative_red).format(format_dict)

#     -------------------------------------------------------------------------------------------------

from math import ceil
import string
import re

class CompanyListReader:
        header = {
                "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.75 Safari/537.36",
                "X-Requested-With": "XMLHttpRequest"
        }
        deprecatedList = ['SMLE3', 'TBLE3', 'VNET3']

        def __init__(self):
                self.dtFrame = self.loadFundamentus()
                #Remove deprecated
                self.dtFrame = self.dtFrame[~self.dtFrame['Paper'].isin(self.deprecatedList)]
        
        def loadInfomoney(self):
                url = 'https://www.infomoney.com.br/minhas-financas/confira-o-cnpj-das-acoes-negociadas-em-bolsa-e-saiba-como-declarar-no-imposto-de-renda/'
                r = requests.get(url, headers=self.header)
                rawTable = pd.read_html(r.text, thousands='.',decimal=',')[0]
                return rawTable

        def loadOceans(self):
                url = 'https://www.oceans14.com.br/acoes/'
                r = requests.get(url, headers=self.header)
                rawTable = pd.read_html(r.text, thousands='.',decimal=',')[0]
                return rawTable

        def loadGuia(self):
                pageAmount = 10
                rawTable = pd.DataFrame()
                url = 'https://www.guiainvest.com.br/lista-acoes/default.aspx?listaacaopage='
                r = requests.get(url, headers=self.header)
                df = pd.read_html(r.text, thousands='.',decimal=',')[0]
                res = re.search('Registros\s\d+\s-\s(\d+)\sde\s(\d+)', df.to_string())
                if (res):
                        pageAmount = ceil(int(res.group(2))/ int(res.group(1)))

                for i in range(1, pageAmount):
                        r = requests.get(url + str(i), headers=self.header)
                        rawTable = rawTable.append(pd.read_html(r.text, thousands='.',decimal=',')[0].drop(['Unnamed: 0', 'Atividade Principal'], axis=1))

                return rawTable.reset_index(drop=True)

        def loadAdvfn(self):
                rawTable = pd.DataFrame()
                url = 'https://br.advfn.com/bolsa-de-valores/bovespa/'
                for pg in string.ascii_uppercase:    
                        r = requests.get(url + pg, headers=self.header)
                        rawTable = rawTable.append(pd.read_html(r.text, thousands='.',decimal=',')[0])

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
                r = requests.get(url, headers=self.header)
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
    prcReader = PriceReader(None, ['EA', 'CCJ'] )
    prcReader.load()
    print(prcReader.getCurrentValue('CCJ', '2018-02-14'))