import datetime as dt
import sys
import threading
import time
from collections import namedtuple

import numpy as np
import pandas as pd

from BroakerParser import ClearDivStatement, ReadOrders, TDAmeritrade
from FinanceTools import (
    Color,
    DividendReader,
    PerformanceBlueprint,
    PerformanceViewer,
    Portifolio,
    PriceReader,
    Profit,
    SplitsReader,
    TableAccumulator,
    YfinanceReader,
)
from IRPF_Tools import *

pd.options.display.float_format = "${:,.2f}".format


currency_market_map = {"us": "$", "br": "R$", "uk": "£"}


class Wallet:
    def __init__(self, work_dir):
        self.work_dir = work_dir
        if work_dir[-1] != "/":
            self.work_dir += "/"
        self.td_config = namedtuple("config", "order_dir dataframe_path dividends_statement_path recommended_wallet")
        self.clear_config = namedtuple("config", "order_dir dataframe_path dividends_statement_path recommended_wallet")

        self.td_config.order_dir = self.work_dir + "Notas_TD"
        self.td_config.dividends_statement_path = self.work_dir + "Notas_TD"
        self.td_config.dataframe_path = self.work_dir + "TD.csv"
        self.td_config.recommended_wallet = self.td_config.order_dir + "/global_wallet.json"

        self.clear_config.order_dir = self.work_dir + "Notas_Clear"
        self.clear_config.dividends_statement_path = self.work_dir + "Notas_Clear/Statements"
        self.clear_config.dataframe_path = self.work_dir + "operations.csv"
        self.clear_config.recommended_wallet = self.clear_config.order_dir + "/iv_wallet.json"
        try:
            os.mkdir("debug")
        except:
            pass

    def convert_table(self):
        file_ = self.work_dir + "/operations.csv"
        df = pd.read_csv(file_)
        df.sort_values(by=["Date", "Type", "Fee"], ascending=[True, False, True], inplace=True)
        df.to_csv(file_, index=False)

    def open_dataframe(self):
        if self.market == "br":
            ReadOrders(self.clear_config.order_dir, self.clear_config.dataframe_path, "Clear")
            self.df = pd.read_csv(self.clear_config.dataframe_path)
        else:
            td = TDAmeritrade(self.td_config.dataframe_path)
            td.read_statement(self.td_config.order_dir)
            self.df = pd.read_csv(self.td_config.dataframe_path)

        self.df = self.df.iloc[:, :7]
        self.df.columns = ["SYMBOL", "DATE", "PRICE", "QUANTITY", "OPERATION", "TYPE", "FEE"]

        # drop empty lines
        self.df = self.df[self.df["DATE"].astype(bool)].dropna()

        if self.market == "br":
            self.brTickers = np.sort(self.df[self.df["TYPE"].isin(["Ação"])]["SYMBOL"].unique()).tolist()
            self.fiiTickers = np.sort(self.df[self.df["TYPE"] == "FII"]["SYMBOL"].unique()).tolist()
            self.usTickers = []
        else:
            self.brTickers = []
            self.fiiTickers = []
            self.usTickers = np.sort(self.df[self.df["TYPE"].isin(["STOCK", "REIT"])]["SYMBOL"].unique()).tolist()

        if self.df["PRICE"].apply(type).eq(str).any():
            self.df["PRICE"] = self.df["PRICE"].str.replace(",", "")
            self.df["PRICE"] = pd.to_numeric(self.df["PRICE"], errors="coerce")
            self.df["QUANTITY"] = pd.to_numeric(self.df["QUANTITY"], errors="coerce")
            self.df["FEE"] = pd.to_numeric(self.df["FEE"], errors="coerce")

        if self.df["DATE"].apply(type).eq(str).any():
            self.df["DATE"] = self.df.DATE.str.replace("-", "/")
            self.df["DATE"] = pd.to_datetime(self.df["DATE"], format="%Y/%m/%d")
        self.df["Year"] = pd.DatetimeIndex(self.df["DATE"]).year
        self.df["Month"] = pd.DatetimeIndex(self.df["DATE"]).month_name()

        # Sort the table by date and Type and reset index numeration
        self.df.sort_values(by=["DATE", "OPERATION"], ascending=[True, True], inplace=True)
        self.df.reset_index(drop=True, inplace=True)

        # turn all sell amount negative
        self.df.loc[self.df.OPERATION == "S", ["QUANTITY"]] *= -1

        # Get the oldest order date
        self.start_date = self.df.iloc[0]["DATE"]
        self.df["AMOUNT"] = self.df["PRICE"] * self.df["QUANTITY"]

    def load_statement(self):
        if self.market == "br":
            st = ClearDivStatement(
                self.clear_config.dividends_statement_path, self.clear_config.dividends_statement_path, "divTable"
            )
            st.process()
            self.divStatement = st.finish()
        else:
            self.divStatement = pd.DataFrame()

    def load_external_data(self):
        start_time = time.time()

        div_start_date = self.start_date
        if not self.divStatement.empty:
            div_start_date = self.divStatement.iloc[-1]["DATE"]

        self.prcReader = PriceReader(self.brTickers + self.fiiTickers, self.usTickers, self.start_date)
        self.splReader = SplitsReader(self.brTickers, self.usTickers, self.start_date)

        if self.market == "br":
            self.divReader = DividendReader(self.brTickers, self.fiiTickers, None, div_start_date)
        else:
            self.divReader = YfinanceReader(None, None, self.usTickers, div_start_date)

        def threadExecutor(obj):
            obj.load()

        threadList = []
        threadList.append(threading.Thread(target=threadExecutor, args=(self.prcReader,)))
        threadList.append(threading.Thread(target=threadExecutor, args=(self.divReader,)))
        threadList.append(threading.Thread(target=threadExecutor, args=(self.splReader,)))

        for td in threadList:
            td.start()

        for td in threadList:
            td.join()

        print("Executed in %s seconds" % (time.time() - start_time))
        self.prcReader.df.to_csv("debug/log_pcr.tsv", sep="\t")

    def load_recommended_wallet(self):
        import json

        wallet_file = self.clear_config.recommended_wallet if self.market == "br" else self.td_config.recommended_wallet
        self.recommended_wallet = None
        with open(wallet_file) as file:
            self.recommended_wallet = json.load(file)

    def merge_statement_data(self):
        if self.market == "br":
            self.df["PAYDATE"] = self.df["DATE"]

            def getType(symbol):
                tmp = self.df[self.df.SYMBOL == symbol]
                if tmp.empty:
                    return symbol
                return tmp.iloc[0]["TYPE"]

            divTable = self.divStatement
            divTable["TYPE"] = divTable["SYMBOL"].map(lambda x: getType(x))
            divTable["FEE"] = 0
            divTable["Year"] = pd.DatetimeIndex(divTable["DATE"]).year
            divTable["Month"] = pd.DatetimeIndex(divTable["DATE"]).month_name()
            divTable["AMOUNT"] = divTable["PRICE"] * divTable["QUANTITY"]
            divTable = divTable.drop(columns="DESCRIPTION")

            self.df = pd.concat([self.df, divTable])

    def merge_external_data(self):
        self.df["acum_qty"] = 0
        self.df["PM"] = 0
        self.df["CASH"] = 0
        self.df["PAYDATE"] = self.df["DATE"]
        today = dt.datetime.today().strftime("%Y-%m-%d")

        for paper in self.brTickers + self.fiiTickers + self.usTickers:
            paperTable = self.df[self.df.SYMBOL == paper]
            fromDate = paperTable.iloc[0]["DATE"]

            divTable = self.divReader.getPeriod(paper, fromDate, today).reset_index()
            if not self.divStatement.empty:
                divTable = divTable[pd.to_datetime(divTable.PAYDATE) > self.divStatement.iloc[-1]["DATE"]]
            divTable["QUANTITY"] = 1
            divTable["TYPE"] = paperTable.iloc[0]["TYPE"]
            divTable["FEE"] = 0
            divTable["Year"] = pd.DatetimeIndex(divTable["DATE"]).year
            divTable["Month"] = pd.DatetimeIndex(divTable["DATE"]).month_name()
            divTable["AMOUNT"] = 0
            divTable["acum_qty"] = 0
            divTable["CASH"] = 0
            self.df = pd.concat([self.df, divTable])

            splitTable = self.splReader.getPeriod(paper, fromDate, today).reset_index()
            splitTable["PRICE"] = 0
            splitTable["OPERATION"] = "SPLIT"
            splitTable["TYPE"] = paperTable.iloc[0]["TYPE"]
            splitTable["FEE"] = 0
            splitTable["Year"] = pd.DatetimeIndex(splitTable["DATE"]).year
            splitTable["Month"] = pd.DatetimeIndex(splitTable["DATE"]).month_name()
            splitTable["AMOUNT"] = 0
            splitTable["acum_qty"] = 0
            splitTable["CASH"] = 0
            splitTable["PAYDATE"] = splitTable["DATE"]
            self.df = pd.concat([self.df, splitTable])

    def compute_average_price(self):
        operation_order_map = {
            "C": 0,
            "W": 0,
            "SPLIT": 0,
            "B": 1,
            "S": 2,
            "D": 3,
            "D1": 3,
            "D2": 3,
            "JCP": 3,
            "JCP1": 3,
            "R": 3,
            "R1": 3,
            "T": 4,
            "T1": 4,
            "A": 5,
            "A1": 5,
            "I": 6,
            "I1": 6,
            "CF": 6,
            "RRV": 6,
        }

        self.df["OPERATION_ORDER"] = self.df["OPERATION"].map(lambda x: operation_order_map.get(x, 100))
        self.df.sort_values(["DATE", "OPERATION_ORDER"], inplace=True)
        self.df = self.df.drop("OPERATION_ORDER", axis=1)

        # Calc the average price and rename the columns names
        self.df = self.df.sort_values(["PAYDATE", "OPERATION"], ascending=[True, False])
        self.df = self.df.apply(TableAccumulator(self.prcReader).Cash, axis=1).reset_index(drop=True)

        self.df = (
            self.df.groupby(["SYMBOL"], group_keys=False)
            .apply(TableAccumulator(self.prcReader).ByGroup)
            .reset_index(drop=True)
        )

    def compute_realized_profit(self):
        profit = Profit()
        tmp = self.df.sort_values(by=["DATE", "OPERATION"], ascending=[True, True])
        tmp.reset_index(drop=True)
        self.df = tmp.groupby(["SYMBOL", "DATE"], group_keys=False).apply(profit.Trade).reset_index(drop=True)
        self.df.sort_values(["PAYDATE", "OPERATION"], ascending=[True, False]).to_csv(
            f"debug/df_log_{self.market}.tsv", sep="\t"
        )

        rl = self.df[self.df.OPERATION == "S"][
            ["DATE", "SYMBOL", "TYPE", "AMOUNT", "Profit", "DayTrade", "Month", "Year"]
        ]
        rl1 = rl[["DATE", "SYMBOL", "TYPE", "AMOUNT", "Profit", "DayTrade"]].copy(deep=True)
        rl1["DATE"] = rl1["DATE"].apply(lambda x: x.strftime("%Y-%m-%d"))
        rl1 = rl1.groupby(["DATE", "SYMBOL", "TYPE"]).sum().reset_index()
        rl1.loc["Total", "Profit"] = rl["Profit"].sum()
        rl1["AMOUNT"] = rl1["AMOUNT"].abs()
        rl1.loc["Total", "AMOUNT"] = 0
        rl1 = rl1.fillna(" ").reset_index(drop=True)
        self.realized_profit_df = rl1.style.applymap(Color().color_negative_red, subset=["Profit", "AMOUNT"]).format(
            {"AMOUNT": f"{self.currency} {{:,.2f}}", "Profit": f"{self.currency} {{:,.2f}}", "DayTrade": "{}"}
        )

        rl1 = rl.groupby("SYMBOL").Profit.sum().reset_index()
        rl1.loc["Total", "Profit"] = rl1["Profit"].sum()
        rl1 = rl1.fillna(" ").reset_index(drop=True)
        self.realized_profit_by_symbol_df = rl1.style.applymap(Color().color_negative_red, subset=["Profit"]).format(
            {"Profit": f"{self.currency} {{:,.2f}}"}
        )

        def Pivot(tb):
            if tb.empty:
                return pd.DataFrame()
            pvt = tb.pivot_table(
                index="Year",
                columns="Month",
                values="Profit",
                margins=True,
                margins_name="Total",
                aggfunc="sum",
                fill_value=0,
            )
            sorted_m = sorted(pvt.columns[:-1], key=lambda month: dt.datetime.strptime(month, "%B"))
            sorted_m.append(pvt.columns[-1])
            pvt = pvt.reindex(sorted_m, axis=1)
            return pvt.style.applymap(Color().color_negative_red).format("{:,.2f}")

        if not rl.empty:
            self.realized_profit_pivot_all = Pivot(rl)
            self.realized_profit_pivot_stock = Pivot(rl[rl["TYPE"] != "FII"])
            self.realized_profit_pivot_fii = Pivot(rl[rl["TYPE"] == "FII"])

    def compute_portifolio(self):
        today = dt.datetime.today().strftime("%Y-%m-%d")
        self.portifolio_df = Portifolio(
            self.prcReader, self.splReader, today, self.df, self.recommended_wallet, self.currency
        ).show()

    def compute_blueprint(self):
        p = PerformanceBlueprint(
            self.prcReader, self.splReader, self.df, dt.datetime.today().strftime("%Y-%m-%d"), currency=self.currency
        )
        self.blueprint_df = PerformanceViewer(p.calc()).show()

    def compute_dividends(self):
        self.prov_month = pd.DataFrame()
        for n in range(1, -1, -1):
            date = dt.datetime.today() - pd.DateOffset(months=n)

            m = int(date.strftime("%m"))
            y = int(date.strftime("%Y"))
            prov_df = self.df[(self.df["PAYDATE"].dt.month == m) & (self.df["PAYDATE"].dt.year == y)]
            if prov_df.empty:
                continue

            prov_month = prov_df[prov_df["OPERATION"].isin("D R JCP A".split())].copy(deep=True)
            if prov_month.empty:
                prov_month = prov_df[prov_df["OPERATION"].isin("D1 R1 JCP1 A1".split())].copy(deep=True)

            if prov_month.empty:
                continue

            prov_month = prov_month[["PAYDATE", "SYMBOL", "AMOUNT"]]
            prov_month.columns = ["DATE", "SYMBOL", self.currency]
            prov_month = prov_month.groupby(["SYMBOL", "DATE"])[self.currency].sum().reset_index()
            prov_month.sort_values("DATE", inplace=True)
            prov_month["DATE"] = prov_month["DATE"].apply(lambda x: x.strftime("%Y-%m-%d"))
            prov_month.loc["Total", self.currency] = prov_month[self.currency].sum()
            prov_month["MONTH"] = date.strftime("%B")
            self.prov_month = pd.concat([self.prov_month, prov_month.fillna(" ").reset_index(drop=True)])

        if not self.prov_month.empty:
            self.prov_month.set_index(["MONTH", "SYMBOL"], inplace=True)

        prov = self.df[self.df["OPERATION"].isin("D1 R1 JCP1 A1".split())]
        if prov.empty:
            prov = self.df[self.df["OPERATION"].isin("D R JCP A".split())]

        if not prov.empty:
            pvt = prov.pivot_table(
                index="Year",
                columns="Month",
                values="AMOUNT",
                margins=True,
                margins_name="Total",
                aggfunc="sum",
                fill_value=0,
            )
            sorted_m = sorted(pvt.columns[:-1], key=lambda month: dt.datetime.strptime(month, "%B"))
            sorted_m.append(pvt.columns[-1])
            self.pvt_div_table = (
                pvt.reindex(sorted_m, axis=1)
                .style.applymap(Color().color_negative_red)
                .format(f"{self.currency} {{:,.2f}}")
            )
        else:
            self.pvt_div_table = pd.DataFrame()

    def compute_history_blueprint(self, period="all"):
        startPlot = self.start_date
        frequency = "SM"

        if period.lower() != "all":
            frequency = "W"
            wishedStart = dt.datetime.today() - pd.DateOffset(years=int(period.split(" ")[0]))
            if pd.to_datetime(startPlot) < pd.to_datetime(wishedStart):
                startPlot = wishedStart.strftime("%Y-%m-%d")

        monthList = pd.date_range(start=startPlot, end=dt.datetime.today(), freq=frequency).format(
            formatter=lambda x: x.strftime("%Y-%m-%d")
        )
        monthList.append(dt.datetime.today().strftime("%Y-%m-%d"))
        performanceList = []
        if period.lower() == "all":
            performanceList.append([startPlot - pd.DateOffset(weeks=2), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

        for month in monthList:
            p = PerformanceBlueprint(self.prcReader, self.splReader, self.df, month).calc()
            performanceList.append(
                [
                    dt.datetime.strptime(p.date, "%Y-%m-%d"),
                    p.equity,
                    p.cost,
                    p.realizedProfit,
                    p.div,
                    p.paperProfit,
                    p.profit,
                    p.profitRate,
                    p.expense,
                    p.ibov,
                    p.sp500,
                    p.cum_cdb,
                ]
            )

        histProfDF = pd.DataFrame(
            performanceList,
            columns=[
                "Date",
                "Equity",
                "Cost",
                "Profit",
                "Div",
                "paperProfit",
                "TotalProfit",
                "%Profit",
                "Expense",
                "%IBOV",
                "%SP500",
                "CDB",
            ],
        )

        if period.lower() != "all":
            histProfDF["%IBOV"] -= histProfDF.iloc[1, "%IBOV"]
            histProfDF["%SP500"] -= histProfDF.iloc[1, "%SP500"]
            histProfDF["%Profit"] -= histProfDF.iloc[1, "%Profit"]
        self.historic_profit_df = histProfDF
        self.history_df_frequency = frequency

    def generate_charts(self):
        import matplotlib.pyplot as plt
        import matplotlib.ticker as mtick

        histProfDF = self.historic_profit_df

        # the width of the bars: can also be len(x) sequence
        width = 2 if self.history_df_frequency == "W" else 5
        shift = pd.Timedelta(width / 2, unit="d")
        fig, ax = plt.subplots(2, 1, figsize=(32, 9), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
        fig.tight_layout()

        ax[0].plot(histProfDF.Date, histProfDF["%IBOV"], label="ibovespa")
        ax[0].plot(histProfDF.Date, histProfDF["%SP500"], label="S&P500")
        ax[0].plot(histProfDF.Date, histProfDF["%Profit"], label="Wallet")
        ax[0].plot(histProfDF.Date, histProfDF["CDB"], label="CDB")

        minTick = min(histProfDF["%IBOV"].min(), histProfDF["%SP500"].min(), histProfDF["%Profit"].min())
        maxTick = max(histProfDF["%IBOV"].max(), histProfDF["%SP500"].max(), histProfDF["%Profit"].max())

        ax[0].set_yticks(np.arange(minTick, maxTick, 0.03))
        ax[0].axhline(y=0, color="k")
        ax[0].grid(True, which="both")
        ax[0].yaxis.set_major_formatter(mtick.PercentFormatter(1))
        ax[0].legend()

        barsDf = histProfDF[:-1]
        ax[1].bar(barsDf.Date - shift, barsDf["Equity"], width, label="Equity")
        ax[1].bar(barsDf.Date - shift, barsDf["Div"], width, bottom=barsDf["Equity"], label="Div")
        ax[1].bar(barsDf.Date - shift, barsDf["Profit"], width, bottom=barsDf["Div"] + barsDf["Equity"], label="Profit")
        ax[1].bar(barsDf.Date + shift, barsDf["Cost"], width, label="Cost")
        ax[1].legend()
        ax[1].set_ylabel(self.currency)

        plt.xticks(barsDf["Date"], rotation=90)
        plt.xlabel("Date")
        plt.ylabel("gain")
        plt.title("Historical profitability")
        self.history_chart = fig
        return plt.show()

    def run(self, market="br"):
        self.market = market
        self.currency = currency_market_map[market]
        pd.options.display.float_format = f"{self.currency} {{:,.2f}}".format
        self.open_dataframe()
        self.load_statement()
        self.load_external_data()
        self.load_recommended_wallet()

        self.merge_statement_data()
        self.merge_external_data()
        self.compute_average_price()
        self.compute_realized_profit()
        self.compute_portifolio()
        self.compute_blueprint()
        self.compute_dividends()
        self.compute_history_blueprint()

    def export_to_excel(self, outfile):
        # Create a Pandas Excel writer using XlsxWriter as the engine.
        writer = pd.ExcelWriter(outfile, engine="xlsxwriter")
        self.blueprint_df.to_excel(writer, sheet_name="blueprint")
        self.portifolio_df.to_excel(writer, sheet_name="portifolio")

        self.realized_profit_pivot_all.to_excel(writer, sheet_name="realized_profit")
        index = len(self.realized_profit_pivot_all.index) + 2

        if len(self.realized_profit_pivot_fii.index) > 0:
            self.realized_profit_pivot_fii.to_excel(writer, sheet_name="realized_profit", startrow=index)
            index += len(self.realized_profit_pivot_fii.index) + 2

        if len(self.realized_profit_pivot_stock.index) > 0:
            self.realized_profit_pivot_stock.to_excel(writer, sheet_name="realized_profit", startrow=index)
            index += len(self.realized_profit_pivot_stock.index) + 2

        self.realized_profit_df.to_excel(writer, sheet_name="realized_profit", startrow=index)
        index += len(self.realized_profit_df.index) + 2

        self.realized_profit_by_symbol_df.to_excel(writer, sheet_name="realized_profit", startrow=index)

        self.prov_month.to_excel(writer, sheet_name="dividends")
        index = len(self.prov_month.index) + 2
        self.pvt_div_table.to_excel(writer, sheet_name="dividends", startrow=index)

        writer.save()


if __name__ == "__main__":
    if sys.platform == "linux":
        root = "/home/doreis/Documents/"
    else:
        root = "d:/"
    root += "Investing/"
    # wallet = Wallet(root, )
    # wallet.run(market='br')

    wallet_us = Wallet(
        root,
    )
    wallet_us.run(market="us")

    # wallet.export_to_excel(root + 'out.xlsx')
    # wallet.generate_charts()
    # wallet.history_chart.savefig(root + 'chart.png')
    print("Finished")
