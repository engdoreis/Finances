import pandas as pd
from .DataFrameMerger import DataFrameMerger
from .Caching import Caching
from data import DataSchema


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
            table[DataSchema.DATE] = pd.to_datetime(table[DataSchema.DATE], format=DataSchema.DATE_FORMAT)
        self.table = table
        return self.table

    def append(self, data):
        self.cache.append(data)

    def merge(self, data, on=[DataSchema.SYMBOL, DataSchema.DATE], sortby=[DataSchema.DATE]):
        if not self.table.empty:
            merger = DataFrameMerger(self.table)
            data = merger.append(data, on=on)
        data = data.sort_values(by=sortby)
        self.cache.save(data)
        return self.load_data()
