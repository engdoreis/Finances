import os.path
import pandas as pd
import datetime as dt


class Caching:
    def __init__(self, filename) -> None:
        if os.path.exists(filename):
            self.dt = pd.read_csv(filename, sep="\t")
        else:
            self.dt = pd.DataFrame()
        self.filename = filename

    def is_updated(self):
        if self.dt.empty:
            return False
        if not os.path.exists(self.filename):
            return False
        filetime = dt.datetime.fromtimestamp(os.path.getctime(self.filename))
        return filetime.date() == dt.datetime.now().date()

    def append(self, data):
        self.dt = pd.concat([self.dt, data], axis=0)
        self.dt.to_csv(self.filename, sep="\t", index=False)

    def save(self, data):
        self.dt = data
        self.dt.to_csv(self.filename, sep="\t", index=False)

    def get_data(self):
        return self.dt


if __name__ == "__main__":
    cache = Caching("debug/caching.tsv")
