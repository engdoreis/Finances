from pandas_datareader import data as web
import pandas as pd
import datetime as dt
import numpy as np

class PriceReader:
  def __init__(self, stockList, startDate='01-01-2018'):
    self.stockList = stockList
    self.startDate = startDate
    self.fillDate = dt.datetime.today().strftime('%m-%d-%Y')

  def load(self):
    self.df = self.readData(self.stockList, self.startDate).reset_index()
    self.df.columns = self.df.columns.str.replace('\.SA','')
    # display(self.df.head())
    self.df = self.df.set_index('Date')

    indexList = ['^BVSP', '^GSPC', 'BRLUSD=X']
    self.brlIndex = self.readIndexData(indexList, self.startDate).reset_index()
    self.brlIndex.columns = ['Date', 'IBOV', 'S&P500', 'USD']
    self.brlIndex = self.brlIndex.set_index('Date')
    # display(self.brlIndex)

  def setFillDate(self, date):
    self.fillDate = date

  def fillCurrentValue(self, row):
    row['Cotacao'] = self.getCurrentValue(row['Ativo'], self.fillDate)
    return row

  def readData(self, code, startDate='01-01-2018'):
    s=[]
    for c in code:
      s.append(c + '.SA')
    # importar dados para o DataFrame
    return web.DataReader(s, data_source='yahoo', start=startDate)['Close']

  def readIndexData(self, code, startDate='01-01-2018'):
    # importar dados para o DataFrame
    return web.DataReader(code, data_source='yahoo', start=startDate)['Close']

  def getHistory(self, code, start='01-01-2018'):
    return self.df.loc[start:][code]

  def getCurrentValue(self, code, date=None):
    if(date == None):
      return self.df.iloc[-1][code]

    available, date = self.checkLastAvailable(self.df, date)
    if available:
      return self.df.loc[date][code] 
    return self.df.iloc[0][code] 

  def getIndexHistory(self, code, end):
    ret = self.brlIndex.loc[:end][code]
    return ret.dropna()

  def getIndexCurrentValue(self, code, date=None):
    if(date == None):
      return self.brlIndex.iloc[-1][code]

    available,date = self.checkLastAvailable(self.brlIndex, date)
    if available:
      return self.brlIndex.loc[date][code]
    return self.brlIndex.iloc[0][code]

  def checkLastAvailable(self, dtframe, loockDate):
    date = pd.to_datetime(loockDate)
    day = pd.Timedelta(1, unit='d')
    #Look for last available date

    while(not date in dtframe.index):
      date = date - day
      if(date < dtframe.index[0]):
        return False,0
    return True,date

#   -------------------------------------------------------------------------------------------------

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
    self.stockList = dataFrame[dataFrame['Categoria'] == 'Stock']['Codigo'].unique()
    self.fiiList = dataFrame[dataFrame['Categoria'] == 'FII']['Codigo'].unique()

  def load(self):
    self.df = self.loadData(self.stockList, self.stockUrl)
    self.df = self.df.append(self.loadData(self.fiiList, self.fiiUrl))
    self.df = self.df.sort_values(by=['Data', 'Codigo'])
    self.df.set_index("Data", inplace = True)
    self.df = self.df[['Codigo', 'Valor', 'Data de Pagamento']]
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
        rawTable.columns = ['Data', 'Tipo', 'Data de Pagamento', 'Valor'] 

      rawTable['Codigo'] = paper
      if('Por quantas ações' in rawTable.columns):
        rawTable['Valor'] /= rawTable['Por quantas ações']
        # Discount a taxe of 15% when is JCP (Juros sobre capital proprio)
        rawTable['Valor'] = np.where(rawTable['Tipo'] == 'DIVIDENDO',  rawTable['Valor'],  rawTable['Valor'] * 0.85 )
        

      rawTable = rawTable[['Codigo', 'Data','Valor', 'Data de Pagamento']]

      rawTable['Data de Pagamento'] = np.where(rawTable['Data de Pagamento'] == '-',\
              rawTable['Data'], rawTable['Data de Pagamento'])
      
      rawTable['Data de Pagamento'] = pd.to_datetime(rawTable['Data de Pagamento'], format='%d/%m/%Y')
      rawTable['Data'] = pd.to_datetime(rawTable['Data'], format='%d/%m/%Y')

      # display(rawTable.tail())
      tb = tb.append(rawTable)
    return tb

  def getPeriod(self, paper, fromDate, toDate):
    filtered = self.df[self.df['Codigo'] == paper].loc[fromDate:toDate]
    return filtered[['Codigo', 'Valor']]
  # display(tb)

#   -------------------------------------------------------------------------------------------------

import yfinance as yf

class YfinanceReader(DividendReader):
  def loadData(self, paperList, baseUrl):
    res = pd.DataFrame()
    for paper in paperList:
      data = pd.DataFrame(yf.Ticker(paper + '.SA').dividends)
      data['Codigo'] = paper
      res = pd.concat([res,data], axis=0)
    res.index.rename('Data', inplace=True)
    res.columns = ['Valor', 'Codigo']
    res['Data de Pagamento'] = 0
    # display(res[res['Codigo'] == 'CIEL3'])
    return res[['Codigo', 'Valor', 'Data de Pagamento']].reset_index()

#   -------------------------------------------------------------------------------------------------

class SplitsReader:
  def __init__(self, dataFrame):
    self.paperList = dataFrame[dataFrame['Categoria'] == 'Stock']['Codigo'].unique()
  
  def load(self):
    self.df = self.loadData(self.paperList)
    self.df = self.df.sort_values(by=['Data', 'Codigo'])
    self.df.set_index("Data", inplace = True)

  def getPeriod(self, paper, fromDate, toDate):
    filtered = self.df[self.df['Codigo'] == paper].loc[fromDate:toDate]
    return filtered[['Codigo', 'Quantidade']]

  def loadData(self, paperList):
    res = pd.DataFrame()
    for paper in paperList:
      # print(paper)
      try:
        data = pd.DataFrame(yf.Ticker(paper + '.SA').splits)
      except:
        continue
      data['Codigo'] = paper
      res = pd.concat([res,data], axis=0)
    res.index.rename('Data', inplace=True)
    res.columns = ['Quantidade', 'Codigo']
    # display(res)
    return res.reset_index()

#   -------------------------------------------------------------------------------------------------

class TableAccumulator:
  def __init__(self):
    self.avr = self.acumQty = self.acumProv=0  

  def ByRow(self, row):
    total = row.loc['Total']
    stType = row.loc['Tipo']
    qty = row.loc['Quantidade']

    if (stType == 'Compra'):
      self.avr = ((self.avr * self.acumQty) + (row.loc['Valor'] * qty + row.loc['Despesas'])) 
      self.acumQty += qty
      self.avr /= self.acumQty

    elif (stType == 'Venda'):
      self.acumQty += qty
      if (self.acumQty == 0):
        self.acumProv = 0

    elif (stType == "Split"):
      self.acumQty *= qty
      self.avr /= qty

    elif (stType == "Proventos"):
      total = np.nan
      row['Quantidade'] = self.acumQty
      if( self.acumQty > 0 ):
        total = row.loc['Valor'] * self.acumQty
        self.acumProv += total
 

    row['Total'] = total
    row['acumProv'] = self.acumProv
    row['acum_qty'] = self.acumQty
    row['PM'] = self.avr
    return row

  def ByGroup(self, group):    
    self.avr = self.acumQty = self.acumProv = 0  
    return group.apply(self.ByRow, axis=1)

#   -------------------------------------------------------------------------------------------------

#Class to calculate the profit or loss considering day trade rules.
class Profit:
  def __init__(self):
    self.pm = self.amount = 0
  
  def DayTrade(self, row):
    profit = 0
    amount = self.amount + row.Quantidade
    if(row.Tipo == "Compra"):
      self.pm = (row.Valor * row.Quantidade) / amount      
    else:
      profit = (self.pm - row.Valor) * row.Quantidade
      amount = self.amount - row.Quantidade

    self.amount = amount
    row['Profit'] = profit
    row['DayTrade'] = 1
    return row
    
  def Trade(self, dayGroup):
    purchaseDf = dayGroup.loc[dayGroup.Tipo == 'Compra']
    sellDf = dayGroup.loc[dayGroup.Tipo == 'Venda']
    
    sellCount = len(sellDf)
    purchaseCount = len(purchaseDf)
    
    if(sellCount == 0):
      dayGroup['Profit'] = dayGroup['DayTrade'] = 0
      return dayGroup
     
    if(purchaseCount == 0):
      dayGroup['Profit'] = ((dayGroup.Valor - dayGroup.PM) * -dayGroup.Quantidade ) - dayGroup.Despesas
      dayGroup['DayTrade'] = 0
      return dayGroup

    # Day trade detected
    # print('Day Trade detected\n', dayGroup)
    self.pm = self.amount = 0
    return dayGroup.apply(self.DayTrade, axis=1)

#   -------------------------------------------------------------------------------------------------

class Portifolio:
  def __init__(self, priceReader, dFrame):
    self.dtframe = dFrame.groupby(['Codigo']).apply(lambda x: x.tail(1) )[['Codigo', 'PM', 'acum_qty', 'acumProv', 'Categoria']]
    self.dtframe.columns = ['Ativo', 'PM', 'Quantidade', 'Proventos', 'Categoria']
    self.dtframe.reset_index(drop=True, inplace=True)
    self.dtframe["Custo"] = self.dtframe.PM * self.dtframe.Quantidade
    self.dtframe = self.dtframe[self.dtframe['Quantidade'] > 0]

    self.dtframe['Cotacao'] = self.dtframe.apply(priceReader.fillCurrentValue, axis=1)['Cotacao']
    self.dtframe['Cotacao'] = self.dtframe['Cotacao'].fillna(self.dtframe['PM'])
    self.dtframe["Valor"] = self.dtframe['Cotacao'] * self.dtframe.Quantidade

    self.dtframe['Rentabilidade'] = self.dtframe['Valor'] - self.dtframe['Custo']
    self.dtframe['Lucro'] = self.dtframe['Rentabilidade'] + self.dtframe['Proventos']
    self.dtframe['%R'] = self.dtframe['Rentabilidade'] / self.dtframe['Custo'] *100
    self.dtframe['%R+d'] = self.dtframe['Lucro'] / self.dtframe['Custo'] * 100
    self.dtframe = self.dtframe.replace([np.inf, -np.inf], np.nan).fillna(0)

    self.dtframe = self.dtframe[['Ativo', 'PM', 'Cotacao', 'Quantidade', 'Custo', 'Valor', 'Proventos', 'Rentabilidade', 'Lucro', '%R', '%R+d']]
    self.dtframe.set_index('Ativo', inplace=True)
    self.format = {'Cotacao': 'R$ {:,.2f}', 'PM': 'R$ {:.2f}', 'Quantidade': '{:>n}', 'Custo': 'R$ {:,.2f}', 'Valor': 'R$ {:,.2f}', 'Proventos': 'R$ {:,.2f}',\
                  'Rentabilidade': 'R$ {:,.2f}', 'Lucro': 'R$ {:,.2f}', '%R': '{:,.2f}%', '%R+d': '{:,.2f}%'}

  def show(self):
    return self.dtframe.style.applymap(color_negative_red).format(self.format)
#   -------------------------------------------------------------------------------------------------
class PerformanceBlueprint:
  def __init__(self, priceReader, dataframe, date):
    self.pcRdr = priceReader
    self.equity = self.cost = self.realizedProfit = self.div = self.paperProfit = self.profit \
    = self.usdIbov = self.ibov = self.sp500 = self.profitRate = self.expense = 0
    self.date = date
    self.df = dataframe[(dataframe['Data'] <= date)]
    # display(self.df)
    if (not self.df.empty):
      priceReader.setFillDate(self.date)
      self.pt = Portifolio(self.pcRdr,self.df)

  def calc(self):
    if (not self.df.empty):
      ptf = self.pt.dtframe
      self.equity      = (ptf['Cotacao'] * ptf['Quantidade']).sum()
      self.cost        = ptf['Custo'].sum()
      self.realizedProfit = self.df.loc[self.df.Tipo == 'Venda', 'Profit'].sum()
      self.div         = self.df[self.df.Tipo == 'Proventos']['Total'].sum()
      self.paperProfit = self.equity -  self.cost
      self.profit      = self.equity -  self.cost +  self.realizedProfit +  self.div
      self.profitRate  = self.profit / self.cost
      indexHistory     = self.pcRdr.getIndexHistory('IBOV',self.date)
      self.ibov        = indexHistory.iloc[-1]/indexHistory.iloc[0] - 1
      indexHistory     = self.pcRdr.getIndexHistory('S&P500', self.date)
      self.sp500       = indexHistory.iloc[-1]/indexHistory.iloc[0] - 1
      self.expense     = self.df.loc[self.df.Tipo == "Compra",'Despesas'].sum()
      return self

#   -------------------------------------------------------------------------------------------------

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
    dayTrade = self.df[(self.df['DayTrade'] == 1) & (self.df['Categoria'] == stockType)]
    dayTrade = dayTrade.groupby(['Codigo', 'Data'])['Profit'].sum().reset_index()
    dayTrade['Year'] = pd.DatetimeIndex(dayTrade['Data']).year
    dayTrade['Month'] = pd.DatetimeIndex(dayTrade['Data']).month_name()
    dayTrade=dayTrade[['Month','Profit','Year']]
    return dayTrade

  def SwingTrade(self, stockType):
    swingTrade = pd.DataFrame(columns=['Month','Profit'])
    #Filter by stockType and get the year list
    typeDF = self.df[(self.df['DayTrade'] == 0) & (self.df['Categoria'] == stockType)]
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

#   -------------------------------------------------------------------------------------------------

class PerformanceViewer:
    def __init__(self, *args):
        self.pf = pd.DataFrame(columns = ['Item', "Value R$", '%C'])
        if (len(args) == 2 and isinstance(args[0], pd.DataFrame)):
            row = args[0].set_index('Date').loc[args[1]]
            self.buildTable(row['Equity'], row['Cost'], row['Expense'], row['paperProfit'], row['Profit'], row['Div'], row['TotalProfit'])
        elif(isinstance(args[0], PerformanceBlueprint)):
            p = args[0]
            self.buildTable(p.equity, p.cost, p.expense, p.paperProfit, p.realizedProfit, p.div, p.profit)

    def buildTable(self, equity, cost, expense, paperProfit, profit, div, totalProfit):
        self.pf.loc[len(self.pf)] = ['Equity          ' , equity,equity / cost]
        self.pf.loc[len(self.pf)] = ['Cost            ' , cost, 1]
        self.pf.loc[len(self.pf)] = ['Expenses        ' , expense, expense/cost]
        self.pf.loc[len(self.pf)] = ['Paper profit    ' , paperProfit, paperProfit / cost]
        self.pf.loc[len(self.pf)] = ['Realized profit ' , profit, profit / cost]
        self.pf.loc[len(self.pf)] = ['Dividends       ' , div, div / cost]
        self.pf.loc[len(self.pf)] = ['Total Profit    ' , totalProfit, totalProfit / cost]
        self.pf.loc[:, '%C'] *= 100
        self.pf.set_index('Item', inplace=True)

    def show(self):
        format_dict = { 'Value R$': ' {:^,.2f}', '%C': ' {:>.1f}%' }
        return self.pf.style.applymap(color_negative_red).format(format_dict)

#   -------------------------------------------------------------------------------------------------

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
#   -------------------------------------------------------------------------------------------------

##Clear operation cost before 2019
def clear2018Cost(row):
  if (row['Data'].year < 2019):
    return 0
  return row['Despesas']

def color_negative_red(val):
    return 'color: %s' % ('red' if val < 0 else 'green')
