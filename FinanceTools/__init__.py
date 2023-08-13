from .Acumulator import *
from .ADVFN_Page import *
from .Caching import *
from .Color import *
from .CompanyListReader import *
from .DataFrameMerger import *
from .DividendReader import *
from .Fundamentus_Page import *
from .PerformanceBlueprint import *
from .PerformanceViewer import *
from .Portifolio import *
from .PriceReader import *
from .Profit import *
from .SplitsReader import *
from .StockInfoCache import *
from .TableAccumulator import *
from .YfinanceReader import *
from .Currency import *

if __name__ == "__main__":

    def clear2018Cost(row):
        if row["DATE"].year < 2019:
            return 0
        return row["FEE"]

    tickers = [
        "ABEV3",
        "BBDC3",
        "BMEB4",
        "CARD3",
        "CIEL3",
        "COGN3",
        "ECOR3",
        "EGIE3",
        "EZTC3",
        "FLRY3",
        "GOLL4",
        "GRND3",
        "HGTX3",
        "ITUB3",
        "KLBN11",
        "LCAM3",
        "MDIA3",
        "MOVI3",
        "MRVE3",
        "OIBR3",
        "PARD3",
        "PETR4",
        "PRIO3",
        "PSSA3",
        "SBFG3",
        "SMLS3",
        "TASA4",
        "TRIS3",
        "VVAR3",
        "WEGE3",
        "XPBR31",
        "BBFI11B",
        "DEVA11",
        "FAMB11B",
        "FIGS11",
        "GTWR11",
        "HGRE11",
        "HGRU11",
        "HSLG11",
        "HSML11",
        "HTMX11",
        "KNSC11",
        "MFII11",
        "MXRF11",
        "RBRF11",
        "RBRY11",
        "RVBI11",
        "SPTW11",
        "VILG11",
        "VISC11",
        "VRTA11",
        "XPCM11",
        "XPLG11",
        "XPML11",
    ]
    tickers_us = ["CSCO", "VZ", "LUMN", "EA", "NEM", "KWEB", "PRIM", "HOLI"]
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
