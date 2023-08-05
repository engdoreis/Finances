import pandas as pd
from .DataFrameMerger import DataFrameMerger
from .Caching import Caching


class StockInfoCache:
    def __init__(self, cache_file):
        self.cache = Caching(cache_file)
        self.load_data()

    def is_updated(self):
        # return False
        return self.cache.is_updated()

    def load_data(self):
        table = self.cache.get_data()
        if not table.empty:
            table["DATE"] = pd.to_datetime(table["DATE"], format="%Y-%m-%d")
        self.table = table
        return self.table

    def append(self, data):
        self.cache.append(data)

    def merge(self, data, on=["SYMBOL", "DATE"], sortby=["DATE"]):
        if not self.table.empty:
            merger = DataFrameMerger(self.table)
            data = merger.append(data, on=on)
        data = data.sort_values(by=sortby)
        self.cache.save(data)
        return self.load_data()
