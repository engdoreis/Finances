from collections import namedtuple
from glob import glob

import pdfplumber

from .Broaker import *
from .Clear import *
from .ClearDivStatement import *
from .ClearProventos import *
from .OrdersReader import *
from .TDAmeritrade import *


def to_float(str, decimal=",", thousand="."):
    return float(str.replace(thousand, "").replace(decimal, "."))


def Clear_DivStatementTest():
    clear = ClearDivStatement("/tmp/clear/", "/tmp/clear/", "proventos")
    clear.process()
    clear.finish()


def TDAmeritradeTest():
    pdf = pdfplumber.open("d:/Investing/Notas_TD/Trade_Confirmations.pdf", password="371")
    pgObj = TDAmeritrade("d:/Investing/", "TD")
    for page in pdf.pages:
        pgObj.process(page)

    pgObj.finish()


def ClearProventosTest():
    pdf = pdfplumber.open(
        "/home/doreis/Documents/IRPF/2022/2022_informes_redimento/clear/37165263802-2021-Proventos.pdf", password="371"
    )
    pgObj = ClearProventos(
        "/home/doreis/Documents/IRPF/2022/2022_informes_redimento/clear/",
        "/home/doreis/Documents/IRPF/2022/2022_informes_redimento/clear/",
        "proventos",
    )
    for page in pdf.pages:
        pgObj.process(page)

    pgObj.finish()


if __name__ == "__main__":
    # TDAmeritradeTest()
    # ClearProventosTest()
    Clear_DivStatementTest()
