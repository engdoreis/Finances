import unittest
from FinanceTools import *
from pandas.testing import assert_frame_equal

class Test_Taxation:
    def __init__(self):
        self.sample_df = pd.read_csv('test/mock_profitLoss.tsv', sep='\t')

        self.ref_fii = pd.read_csv('test/taxation_ref.tsv', sep='\t')
        self.ref_fii.set_index(["Year", 'Month'], inplace=True)

        self.ref_stock = pd.read_csv('test/taxation_ref_stocks.tsv', sep='\t')
        self.ref_stock.set_index(["Year", 'Month'], inplace=True)

        self.ref_stock_daytrade = pd.read_csv('test/taxation_ref_stocks_daytrade.tsv', sep='\t')
        self.ref_stock_daytrade.set_index(["Year", 'Month'], inplace=True)

        self.tx = Taxation(self.sample_df)
    
    def sanitize_type(self, df):
        dict_columns_type = {'Profit': float,
            'AcumLoss': float,
            'Dutiable': float,
            'Tax': float,
            }
        return df.astype(dict_columns_type)
        
    def process_fii(self):
        self.tx.Process('FII')
        # self.tx.swingTradeTable.to_csv('test/taxation_ref.tsv', sep='\t')
        
        df = self.sanitize_type(self.tx.swingTradeTable)
        # print(self.ref_fii)
        # print(df)
        assert_frame_equal(df, self.ref_fii)

        assert hasattr(self.tx, 'dayTradeTable') ==  False
    
    def process_stocks(self):
        self.tx.Process('Ação')

        df = self.sanitize_type(self.tx.swingTradeTable)
        # df.to_csv('test/taxation_ref_stocks.tsv', sep='\t')
        # print(self.ref_stock)
        # print(df)
        assert_frame_equal(df, self.ref_stock)
        
        df = self.sanitize_type(self.tx.dayTradeTable)
        # df.to_csv('test/taxation_ref_stocks_daytrade.tsv', sep='\t')
        # print(self.ref_stock_daytrade)
        # print(df)
        assert_frame_equal(df, self.ref_stock_daytrade)    

test = Test_Taxation()

test.process_fii()
test.process_stocks()

print("Test ok!")