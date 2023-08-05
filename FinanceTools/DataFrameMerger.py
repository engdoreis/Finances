import pandas as pd


class DataFrameMerger:
    def __init__(self, df):
        self.df = df

    def append(self, new, on):
        columns = self.df.columns.to_list()
        # print(self.df.sort_values(by=['SYMBOL', 'DATE']).reset_index(drop=True))
        # print(new.sort_values(by=['SYMBOL', 'DATE']).reset_index(drop=True))
        self.df = self.df.merge(new, how="outer", on=on, suffixes=["", "_"], indicator=True)
        # print(self.df)
        right_cols = [col for col in self.df.columns if col.endswith("_")]
        for col in right_cols:
            self.df[col[:-1]] = self.df[col[:-1]].fillna(self.df[col])
        self.df = self.df[columns]
        self.df = self.df.drop_duplicates()
        # print(self.df.sort_values(by=['SYMBOL', 'DATE']).reset_index(drop=True))
        return self.df
