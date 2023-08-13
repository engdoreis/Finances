import os
import sys
from glob import glob
from multiprocessing import Process
from shutil import rmtree

import pandas as pd
import pdfplumber

from BroakerParser import Clear, TDAmeritrade
from FinanceTools import CompanyListReader


class OrderOrganizer:
    def __init__(self, inDir):
        self.dtFrame = pd.DataFrame(
            columns=["Code", "Date", "Company", "Type", "Category", "Qty", "Value", "Total", "Sub"]
        )

        files = sorted(glob(inDir + "/*.csv"))
        for file in files:
            self.dtFrame = self.dtFrame.merge(pd.read_csv(file), how="outer")

    def partial_match(self, row):
        code = row["Code"]
        if "FII" in row["Category"]:
            row["Paper"] = code
        else:
            code_number = 0
            if "PN" in code:
                code_number = 4
            elif "ON" in code:
                code_number = 3
            elif "UNT" in code:
                code_number = 11
            else:
                raise f"Unknown stock type: {code}"

            row["Paper"] = str(row["CODE"]) + str(code_number)

        return row

    def finish(self, cmpMap):
        if self.dtFrame["Code"].str.contains("ON|PN|UNT").any():
            self.cmpMap = cmpMap
            self.dtFrame = self.dtFrame.merge(self.cmpMap, how="left", left_on="Company", right_on="NAME")
            self.dtFrame = self.dtFrame.apply(self.partial_match, axis=1).reset_index(drop=True)
            self.dtFrame["Date"] = pd.to_datetime(self.dtFrame["Date"])
            self.dtFrame = self.dtFrame.sort_values("Date").reset_index(drop=True)
        else:
            self.dtFrame["Paper"] = self.dtFrame["Code"]
        return self.dtFrame


def ReadPages(file, dir_, pdf_type="Clear"):
    pdf = pdfplumber.open(file, password="371")

    filename = os.path.basename(file).split(".")[0] + ".csv"
    pgObj = Clear(dir_, filename) if pdf_type == "Clear" else TDAmeritrade(dir_, filename)

    for page in pdf.pages:
        pgObj.process(page)

    pgObj.finish()


def ReadOrders(indir="Notas_Clear", out_file="operations.csv", pdf_type="Clear"):
    input_dir = indir
    output_dir = indir + "/.."
    tmp_dir = output_dir + "/tmpDir"

    if os.path.exists(tmp_dir):
        rmtree(tmp_dir)
    os.mkdir(tmp_dir)

    files = sorted(glob(input_dir + "/*.pdf"))

    processes = []

    print("Starting pages")
    for file in files:
        processes.append(
            Process(
                target=ReadPages,
                args=(
                    file,
                    tmp_dir,
                    pdf_type,
                ),
            )
        )

    for pcs in processes:
        pcs.start()

    print("Getting tickers names...", end="\r")

    companyListReader = CompanyListReader()
    print("Getting tickers names...Done")

    for pcs in processes:
        pcs.join()

    print("Pages done")
    print("Tickers merging...", end="\r")
    companyMap = companyListReader.dtFrame
    filename = output_dir + "/map.csv"
    if companyMap.empty:
        companyMap = pd.read_csv(filename)
    else:
        companyMap.to_csv(filename)

    oOrg = OrderOrganizer(tmp_dir)
    oOrg.finish(companyMap)

    print("Tickers merging...Done")

    tempFile = out_file + ".tmp"
    oOrg.dtFrame[["Paper", "Date", "Value", "Qty", "Type", "Category", "Fee", "Company"]].to_csv(tempFile, index=False)

    try:
        dtypes = {"Qty": float, "Value": float}
        existentDF = pd.read_csv(out_file, dtype=dtypes)
        outDF = pd.read_csv(tempFile, dtype=dtypes)
        existentDF = existentDF[existentDF["Date"].astype(bool)].dropna()

        diff = outDF.merge(
            existentDF, how="outer", on=["Date", "Value", "Qty", "Fee", "Company"], suffixes=["", "_"], indicator=True
        )
        diff = diff[diff["_merge"] == "left_only"]
        diff = diff.iloc[:, :8]

        new = pd.concat([existentDF, diff])
        new["Qty"] = new["Qty"].astype(float).round(6)
        new.to_csv(out_file, index=False)
        os.remove(tempFile)
    except:
        os.rename(tempFile, out_file)


if __name__ == "__main__":
    # start_time = time.time()
    indir = "d:/Investing/Notas_Clear"
    outfile = "operations.csv"
    pdfType = "Clear"
    if len(sys.argv) > 2:
        indir = str(sys.argv[1])
        outfile = str(sys.argv[2])
        pdfType = str(sys.argv[3])

    ReadOrders(indir, outfile, pdfType)

    # print("--- %s seconds ---" % (time.time() - start_time))
