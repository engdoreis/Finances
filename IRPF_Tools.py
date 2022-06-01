import os.path
from unicodedata import decimal
from pandas_datareader import data as web
import pandas as pd
import datetime as dt
import numpy as np
from Caching import *
from FinanceTools import *
import requests
http_header = {
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.75 Safari/537.36",
        "X-Requested-With": "XMLHttpRequest"
    }

class ProcessedOrders():
    def __init__(self, file):
        self.dFrame = pd.read_csv(file, sep='\t')
        self.dFrame['DATE'] = pd.to_datetime(self.dFrame['DATE'], format='%Y/%m/%d')

    def import_df(self):
        return self.dFrame
class Taxation:
    def __init__(self, file, stockTaxFreeMonth=20000, stockTaxRate=0.2, fiiTaxRate=0.2, daytradeTaxRate=0.2):
        self.df = ProcessedOrders(file).import_df()
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

        if(not self.df['TYPE'].str.contains(stockType).any()):
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

#-------------------------------------------------------------------------------------------------
class IRPF_BensDireitos:
    def __init__(self, file, cache='debug/CNPJ_caching.tsv'):
        dFrame = ProcessedOrders(file).import_df()
        dFrame = dFrame[dFrame['OPERATION'].isin(['B', 'S', 'SPLIT', 'C'])]

        self.dtframe = pd.DataFrame()
        for year in dFrame['DATE'].dt.year.unique():

            tmp = dFrame[dFrame['DATE'] < f'{year}-12-31']

            tmp.sort_values(['PAYDATE', 'OPERATION'], ascending=[True, False],inplace=True)
            tmp=tmp.apply(TableAccumulator().Cash, axis=1)
            cash = tmp.iloc[-1]['CASH']
            cash_brl = tmp.iloc[-1]['PM_BRL']
            tmp =  tmp[ tmp['SYMBOL'] != 'CASH']

            tmp = tmp.groupby(['SYMBOL']).apply(lambda x: x.tail(1))
            tmp = tmp[['SYMBOL', 'PM', 'acum_qty', 'acumProv', 'PM_BRL']]
            # print(tmp)
            tmp.columns = ['SYMBOL', 'COST', 'QUANTITY', 'DIVIDENDS', 'COST_BRL']
            tmp['COST'] *= tmp['QUANTITY']
            tmp['COST_BRL'] *= tmp['QUANTITY']
            tmp.reset_index(inplace=True, drop=True )
            # print(tmp)
            tmp = tmp[tmp['QUANTITY'] > 0]

            newLine = {'SYMBOL':'CASH', 'COST':cash, 'QUANTITY':1, 'DIVIDENDS':0, 'COST_BRL':cash_brl}
            tmp = pd.concat([tmp, pd.DataFrame(newLine, index=[0])])

            tmp = tmp[['SYMBOL', 'QUANTITY', 'COST', 'COST_BRL']].set_index('SYMBOL')

            tmp.columns = pd.MultiIndex.from_product([[f'{year}-12-31'],tmp.columns])
            self.dtframe = pd.concat([self.dtframe, tmp], axis=1)
            # print(self.dtframe)
        self.dtframe = self.dtframe.replace([np.inf, -np.inf], np.nan).fillna(0)
        self.ticker_list = dFrame.drop_duplicates('SYMBOL')[['SYMBOL', 'TYPE']]
        self.ticker_list = self.ticker_list[self.ticker_list['SYMBOL'].isin(self.dtframe.index)]
        self.cache = Caching(cache)

    def load(self):
        for index, type in zip(self.ticker_list['SYMBOL'], self.ticker_list['TYPE']):
            self.dtframe.loc[index,'CNPJ'] = self.get_cnpj(index, type)
            self.dtframe.loc[index,'DESC'] = 'Corretora ' + ('TD Ametridade' if type == 'STOCK' else 'Clear') + f' - {type} - {index} x '
    
    def filter_by_year(self, year):
        from_date = f'{year-1}-12-31'
        to_date = f'{year}-12-31'
        if not from_date in self.dtframe:
            res = self.dtframe[[to_date, 'CNPJ', 'DESC']]
        else:
            res = self.dtframe[[from_date, to_date, 'CNPJ', 'DESC']]

        res = res[(res[to_date, 'QUANTITY'] > 0) | (res[from_date, 'QUANTITY'] > 0 )]

        currency = "BRL"
        if (res[to_date, 'COST'] != res[to_date, 'COST_BRL']).any():
            currency = "USD" 
            res.columns.drop(['CNPJ'])

        res['DESC'] += res[to_date, 'QUANTITY'].astype(int).astype(str) + f' = {currency} ' + res[to_date, 'COST'].round(2).astype(str)
        return res
    
    def get_cnpj_from_cache(self, symbol):
        tmp_df = self.cache.get_data()
        if tmp_df.empty:
            return None
        if not tmp_df['SYMBOL'].isin([symbol]).any():
            return None
        return tmp_df[tmp_df['SYMBOL'] == symbol].iloc[0]['CNPJ']
    
    def add_cnpj_to_cache(self, symbol, cnpj, name, type):
        self.cache.append(pd.DataFrame({'SYMBOL': [symbol], 'CNPJ': [cnpj], 'NAME': [name], 'TYPE': [type]}))

    def get_last_year(self, exchange_rate=1.0):
        return self.filter_by_year(int(dt.datetime.today().strftime("%Y")) - 1)
    
    def get_cnpj(self, ticker, type):
        if type == 'STOCK':
            return 'No Data'

        cnpj = self.get_cnpj_from_cache(ticker)
        if cnpj is None:
            try:
                if type == 'FII':
                    cnpj = self.get_cnpj_fii(ticker)
                else:
                    cnpj = self.get_cnpj_stock(ticker)
            except:
                return 'Page no found'

            if self.is_cnpj_valid(cnpj):
                self.add_cnpj_to_cache(ticker, cnpj, 'Unknown', type)

        return cnpj
        
    def get_cnpj_fii(self, ticker):
        url = f'https://www.fundamentus.com.br/fii_administrador.php?papel={ticker}'
        r = requests.get(url, headers=http_header)
        rawTable = pd.read_html(r.text, thousands='.',decimal=',')[0].fillna('Unknown')[[0,1]]
        return rawTable.loc[2][1]
            
    def get_cnpj_stock(self, ticker):
        url = f'https://br.advfn.com/bolsa-de-valores/bovespa/{ticker}/empresa'
        r = requests.get(url, headers=http_header)
        rawTable = pd.read_html(r.text, thousands='.',decimal=',')[0].fillna('Unknown')[[0,1]]
        res = rawTable.loc[1][1]
        return f'{res[:2]}.{res[2:5]}.{res[5:8]}/{res[8:12]}-{res[12:]}'

    def show(self, fdf=None):
        if fdf is None:
            fdf = self.dtframe
        if fdf.empty:
            return

        years = fdf.columns.get_level_values(0).unique()[:-1]
        format = {}
        for year in years:
            case = {(year, 'QUANTITY'): "{:>n}", (year, 'COST'): "{:.2f}", (year, 'COST_BRL'): "{:.2f}" }
            format.update(case)

        return fdf.style.format(format, decimal=',')
    def is_cnpj_valid(self, cnpj):
        return cnpj[0].isdigit()
#     -------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    irpf = IRPF_BensDireitos('debug/df_log.tsv')
    irpf.load()
    print(irpf.get_last_year())

