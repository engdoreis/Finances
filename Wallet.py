import datetime as dt
import sys
import threading
import time
from enum import Enum

import numpy as np
import pandas as pd
from data import DataSchema

from BroakerParser import ClearDivStatement, ReadOrders, TDAmeritrade, Trading212, CharlesChwab
from FinanceTools import (
    Color,
    DividendReader,
    PerformanceSnapshot,
    PerformanceViewer,
    Portfolio,
    PriceReader,
    Profit,
    SplitsReader,
    TableAccumulator,
    YfinanceReader,
)
from IRPF_Tools import *

pd.options.display.float_format = "${:,.2f}".format


@dataclass
class Currency:
    name: str
    symbol: str


class Broker(Enum):
    CLEAR = 1
    TDAMERITRADE = 2
    TRADING212 = 3
    CHARLES_SCHWAB = 4


currency_market_map = {
    Broker.CHARLES_SCHWAB: Currency("USD", "$"),
    Broker.TDAMERITRADE: Currency("USD", "$"),
    Broker.CLEAR: Currency("BRL", "R$"),
    Broker.TRADING212: Currency("GBP", "£"),
}


@dataclass
class Input:
    broker: Broker
    statement_dir: str
    recommended_wallet: str = None


class Wallet:
    input = None

    def __init__(self, work_dir: str):
        self.work_dir = work_dir
        if work_dir[-1] != "/":
            self.work_dir += "/"

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
        if self.input.broker == Broker.CLEAR:
            ReadOrders(self.input.statement_dir, self.clear_config.dataframe_path, "Clear")
            self.df = pd.read_csv(self.clear_config.dataframe_path)
        elif self.input.broker == Broker.TDAMERITRADE:
            csv = self.work_dir + "TD.csv"
            broker = TDAmeritrade(csv)
            broker.read_statement(self.input.statement_dir)
            self.df = pd.read_csv(csv)
        elif self.input.broker == Broker.CHARLES_SCHWAB:
            csv = self.work_dir + "SCHWAB.csv"
            broker = CharlesChwab(csv)
            broker.read_statement(self.input.statement_dir)
            self.df = pd.read_csv(csv)
        elif self.input.broker == Broker.TRADING212:
            csv = self.work_dir + "T212.csv"
            broker = Trading212(csv)
            broker.read_statement(self.input.statement_dir)
            self.df = pd.read_csv(csv)

        DataSchema.assert_base_columns(self.df)
        self.df = self.df[DataSchema.base_columns()]

        # drop empty lines
        self.df = self.df[self.df[DataSchema.DATE].astype(bool)].dropna()

        if self.input.broker == Broker.CLEAR:
            self.brTickers = np.sort(
                self.df[self.df[DataSchema.TYPE].isin(["Ação"])][DataSchema.SYMBOL].unique()
            ).tolist()
            self.fiiTickers = np.sort(self.df[self.df[DataSchema.TYPE] == "FII"][DataSchema.SYMBOL].unique()).tolist()
            self.usTickers = []
        else:
            self.brTickers = []
            self.fiiTickers = []
            self.usTickers = np.sort(
                self.df[self.df[DataSchema.TYPE].isin(["STOCK", "REIT"])][DataSchema.SYMBOL].unique()
            ).tolist()

        if self.df[DataSchema.PRICE].apply(type).eq(str).any():
            self.df[DataSchema.PRICE] = self.df[DataSchema.PRICE].str.replace(",", "")
            self.df[DataSchema.PRICE] = pd.to_numeric(self.df[DataSchema.PRICE], errors="coerce")
            self.df[DataSchema.QTY] = pd.to_numeric(self.df[DataSchema.QTY], errors="coerce")
            self.df[DataSchema.FEES] = pd.to_numeric(self.df[DataSchema.FEES], errors="coerce")

        if self.df[DataSchema.DATE].apply(type).eq(str).any():
            self.df[DataSchema.DATE] = self.df.DATE.str.replace("-", "/")
            self.df[DataSchema.DATE] = pd.to_datetime(self.df[DataSchema.DATE], format="%Y/%m/%d")
        self.df[DataSchema.YEAR] = pd.DatetimeIndex(self.df[DataSchema.DATE]).year
        self.df[DataSchema.MONTH] = pd.DatetimeIndex(self.df[DataSchema.DATE]).month_name()

        # Sort the table by date and Type and reset index numeration
        self.df.sort_values(by=[DataSchema.DATE, DataSchema.OPERATION], ascending=[True, True], inplace=True)
        self.df.reset_index(drop=True, inplace=True)

        # turn all sell amount negative
        self.df.loc[self.df.OPERATION == "S", [DataSchema.QTY]] *= -1

        # Get the oldest order date
        self.start_date = self.df.iloc[0][DataSchema.DATE]
        self.df[DataSchema.AMOUNT] = self.df[DataSchema.PRICE] * self.df[DataSchema.QTY]

    def load_statement(self):
        if self.input.broker == Broker.CLEAR:
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
            div_start_date = self.divStatement.iloc[-1][DataSchema.DATE]

        self.prcReader = PriceReader(self.brTickers + self.fiiTickers, self.usTickers, self.start_date)
        self.splReader = SplitsReader(self.brTickers, self.usTickers, self.start_date)

        if self.input.broker == Broker.CLEAR:
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

        self.recommended_wallet = None
        if self.input.recommended_wallet == None:
            return

        wallet_file = self.input.recommended_wallet if self.input.broker == "clear" else self.input.recommended_wallet
        self.recommended_wallet = None
        with open(wallet_file) as file:
            self.recommended_wallet = json.load(file)

    def merge_statement_data(self):
        if self.input.broker == Broker.CLEAR:
            self.df[DataSchema.PAYDATE] = self.df[DataSchema.DATE]

            def getType(symbol):
                tmp = self.df[self.df[DataSchema.SYMBOL] == symbol]
                if tmp.empty:
                    return symbol
                return tmp.iloc[0][DataSchema.TYPE]

            divTable = self.divStatement
            divTable[DataSchema.TYPE] = divTable[DataSchema.SYMBOL].map(lambda x: getType(x))
            divTable[DataSchema.FEES] = 0
            divTable[DataSchema.YEAR] = pd.DatetimeIndex(divTable[DataSchema.DATE]).year
            divTable[DataSchema.MONTH] = pd.DatetimeIndex(divTable[DataSchema.DATE]).month_name()
            divTable[DataSchema.AMOUNT] = divTable[DataSchema.PRICE] * divTable[DataSchema.QTY]
            divTable = divTable.drop(columns=DataSchema.DESCRIPTION)

            self.df = pd.concat([self.df, divTable])

    def merge_external_data(self):
        self.df[DataSchema.QTY_ACUM] = 0
        self.df[DataSchema.AVERAGE_PRICE] = 0
        self.df[DataSchema.CASH] = 0
        self.df[DataSchema.PAYDATE] = self.df[DataSchema.DATE]
        today = dt.datetime.today().strftime("%Y-%m-%d")

        for paper in self.brTickers + self.fiiTickers + self.usTickers:
            paperTable = self.df[self.df[DataSchema.SYMBOL] == paper]
            fromDate = paperTable.iloc[0][DataSchema.DATE]

            divTable = self.divReader.getPeriod(paper, fromDate, today).reset_index()
            if not self.divStatement.empty:
                divTable = divTable[
                    pd.to_datetime(divTable[DataSchema.PAYDATE]) > self.divStatement.iloc[-1][DataSchema.DATE]
                ]
            divTable[DataSchema.QTY] = 1
            divTable[DataSchema.TYPE] = paperTable.iloc[0][DataSchema.TYPE]
            divTable[DataSchema.FEES] = 0
            divTable[DataSchema.YEAR] = pd.DatetimeIndex(divTable[DataSchema.DATE]).year
            divTable[DataSchema.MONTH] = pd.DatetimeIndex(divTable[DataSchema.DATE]).month_name()
            divTable[DataSchema.AMOUNT] = 0
            divTable[DataSchema.QTY_ACUM] = 0
            divTable[DataSchema.CASH] = 0
            self.df = pd.concat([self.df, divTable])

            splitTable = self.splReader.getPeriod(paper, fromDate, today).reset_index()
            splitTable[DataSchema.PRICE] = 0
            splitTable[DataSchema.OPERATION] = "SPLIT"
            splitTable[DataSchema.TYPE] = paperTable.iloc[0][DataSchema.TYPE]
            splitTable[DataSchema.FEES] = 0
            splitTable[DataSchema.YEAR] = pd.DatetimeIndex(splitTable[DataSchema.DATE]).year
            splitTable[DataSchema.MONTH] = pd.DatetimeIndex(splitTable[DataSchema.DATE]).month_name()
            splitTable[DataSchema.AMOUNT] = 0
            splitTable[DataSchema.QTY_ACUM] = 0
            splitTable[DataSchema.CASH] = 0
            splitTable[DataSchema.PAYDATE] = splitTable[DataSchema.DATE]
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

        self.df["OPERATION_ORDER"] = self.df[DataSchema.OPERATION].map(lambda x: operation_order_map.get(x, 100))
        self.df.sort_values([DataSchema.DATE, "OPERATION_ORDER"], inplace=True)
        self.df = self.df.drop("OPERATION_ORDER", axis=1)

        # Calc the average price and rename the columns names
        self.df = self.df.sort_values([DataSchema.PAYDATE, DataSchema.OPERATION], ascending=[True, False])
        tab_accum = TableAccumulator(self.prcReader, self.currency.name)
        self.df = self.df.apply(tab_accum.Cash, axis=1).reset_index(drop=True)

        self.df = self.df.groupby([DataSchema.SYMBOL], group_keys=False).apply(tab_accum.ByGroup).reset_index(drop=True)

    def compute_realized_profit(self):
        profit = Profit()
        tmp = self.df.sort_values(by=[DataSchema.DATE, DataSchema.OPERATION], ascending=[True, True])
        tmp.reset_index(drop=True)
        self.df = (
            tmp.groupby([DataSchema.SYMBOL, DataSchema.DATE], group_keys=False)
            .apply(profit.Trade)
            .reset_index(drop=True)
        )
        self.df.sort_values([DataSchema.PAYDATE, DataSchema.OPERATION], ascending=[True, False]).to_csv(
            f"debug/df_log_{self.input.broker}.tsv", sep="\t"
        )

        rl = self.df[self.df.OPERATION == "S"][
            [
                DataSchema.DATE,
                DataSchema.SYMBOL,
                DataSchema.TYPE,
                DataSchema.AMOUNT,
                DataSchema.PROFIT,
                DataSchema.DAYTRADE,
                DataSchema.MONTH,
                DataSchema.YEAR,
            ]
        ]
        rl1 = rl[
            [
                DataSchema.DATE,
                DataSchema.SYMBOL,
                DataSchema.TYPE,
                DataSchema.AMOUNT,
                DataSchema.PROFIT,
                DataSchema.DAYTRADE,
            ]
        ].copy(deep=True)
        rl1[DataSchema.DATE] = rl1[DataSchema.DATE].apply(lambda x: x.strftime("%Y-%m-%d"))
        rl1 = rl1.groupby([DataSchema.DATE, DataSchema.SYMBOL, DataSchema.TYPE]).sum().reset_index()
        rl1.loc["Total", DataSchema.PROFIT] = rl[DataSchema.PROFIT].sum()
        rl1[DataSchema.AMOUNT] = rl1[DataSchema.AMOUNT].abs()
        rl1.loc["Total", DataSchema.AMOUNT] = 0
        rl1 = rl1.fillna(" ").reset_index(drop=True)
        self.realized_profit_df = rl1.style.applymap(
            Color().color_negative_red, subset=[DataSchema.PROFIT, DataSchema.AMOUNT]
        ).format(
            {
                DataSchema.AMOUNT: f"{self.currency.symbol} {{:,.2f}}",
                DataSchema.PROFIT: f"{self.currency.symbol} {{:,.2f}}",
                DataSchema.DAYTRADE: "{}",
            }
        )

        rl1 = rl.groupby(DataSchema.SYMBOL).Profit.sum().reset_index()
        rl1.loc["Total", DataSchema.PROFIT] = rl1[DataSchema.PROFIT].sum()
        rl1 = rl1.fillna(" ").reset_index(drop=True)
        self.realized_profit_by_symbol_df = rl1.style.applymap(
            Color().color_negative_red, subset=[DataSchema.PROFIT]
        ).format({DataSchema.PROFIT: f"{self.currency.symbol} {{:,.2f}}"})

        def Pivot(tb):
            if tb.empty:
                return pd.DataFrame()
            pvt = tb.pivot_table(
                index=DataSchema.YEAR,
                columns=DataSchema.MONTH,
                values=DataSchema.PROFIT,
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
            self.realized_profit_pivot_stock = Pivot(rl[rl[DataSchema.TYPE] != "FII"])
            self.realized_profit_pivot_fii = Pivot(rl[rl[DataSchema.TYPE] == "FII"])

    def compute_portfolio(self):
        today = dt.datetime.today().strftime("%Y-%m-%d")
        portfolio = Portfolio(
            self.prcReader, self.splReader, today, self.df, self.recommended_wallet, self.currency.symbol
        )
        self.portfolio_df = portfolio.get_table()
        self.portfolio_view = portfolio.show()

    def compute_snapshot(self):
        p = PerformanceSnapshot(
            self.prcReader,
            self.splReader,
            self.df,
            dt.datetime.today().strftime("%Y-%m-%d"),
            currency=self.currency.name,
        )
        snapshot = PerformanceViewer(p.calc())
        self.performance_snapshot = snapshot.get_table()
        self.snapshot_view = snapshot.get_formatted()

    def compute_dividends(self):
        self.prov_month = pd.DataFrame()
        for n in range(1, -1, -1):
            date = dt.datetime.today() - pd.DateOffset(months=n)

            m = int(date.strftime("%m"))
            y = int(date.strftime("%Y"))
            prov_df = self.df[(self.df[DataSchema.PAYDATE].dt.month == m) & (self.df[DataSchema.PAYDATE].dt.year == y)]
            if prov_df.empty:
                continue

            prov_month = prov_df[prov_df[DataSchema.OPERATION].isin("D R JCP A".split())].copy(deep=True)
            if prov_month.empty:
                prov_month = prov_df[prov_df[DataSchema.OPERATION].isin("D1 R1 JCP1 A1".split())].copy(deep=True)

            if prov_month.empty:
                continue

            prov_month = prov_month[[DataSchema.PAYDATE, DataSchema.SYMBOL, DataSchema.AMOUNT]]
            prov_month.columns = [DataSchema.DATE, DataSchema.SYMBOL, self.currency.name]
            prov_month = (
                prov_month.groupby([DataSchema.SYMBOL, DataSchema.DATE])[self.currency.name].sum().reset_index()
            )
            prov_month.sort_values(DataSchema.DATE, inplace=True)
            prov_month[DataSchema.DATE] = prov_month[DataSchema.DATE].apply(lambda x: x.strftime("%Y-%m-%d"))
            prov_month.loc["Total", self.currency.name] = prov_month[self.currency.name].sum()
            prov_month[DataSchema.MONTH] = date.strftime("%B")
            self.prov_month = pd.concat([self.prov_month, prov_month.fillna(" ").reset_index(drop=True)])

        if not self.prov_month.empty:
            self.prov_month.set_index([DataSchema.MONTH, DataSchema.SYMBOL], inplace=True)

        prov = self.df[self.df[DataSchema.OPERATION].isin("D1 R1 JCP1 A1".split())]
        if prov.empty:
            prov = self.df[self.df[DataSchema.OPERATION].isin("D R JCP A".split())]

        if not prov.empty:
            pvt = prov.pivot_table(
                index=DataSchema.YEAR,
                columns=DataSchema.MONTH,
                values=DataSchema.AMOUNT,
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
                .format(f"{self.currency.symbol} {{:,.2f}}")
            )
        else:
            self.pvt_div_table = pd.DataFrame()

    def compute_history_snapshot(self, period="all"):
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
            p = PerformanceSnapshot(self.prcReader, self.splReader, self.df, month).calc()
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
                DataSchema.PROFIT,
                "Div",
                "paperProfit",
                "TotalProfit",
                "profit_growth",
                "Expense",
                "ibov_growth",
                "sp500_growth",
                "CDB",
            ],
        )

        if period.lower() != "all":
            histProfDF["ibov_growth"] -= histProfDF.iloc[1, "ibov_growth"]
            histProfDF["sp500_growth"] -= histProfDF.iloc[1, "sp500_growth"]
            histProfDF["profit_growth"] -= histProfDF.iloc[1, "profit_growth"]
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

        ax[0].plot(histProfDF.Date, histProfDF["ibov_growth"], label="ibovespa")
        ax[0].plot(histProfDF.Date, histProfDF["sp500_growth"], label="S&P500")
        ax[0].plot(histProfDF.Date, histProfDF["profit_growth"], label="Wallet")
        ax[0].plot(histProfDF.Date, histProfDF["CDB"], label="CDB")

        minTick = min(
            histProfDF["ibov_growth"].min(), histProfDF["sp500_growth"].min(), histProfDF["profit_growth"].min()
        )
        maxTick = max(
            histProfDF["ibov_growth"].max(), histProfDF["sp500_growth"].max(), histProfDF["profit_growth"].max()
        )

        ax[0].set_yticks(np.arange(minTick, maxTick, 0.03))
        ax[0].axhline(y=0, color="k")
        ax[0].grid(True, which="both")
        ax[0].yaxis.set_major_formatter(mtick.PercentFormatter(1))
        ax[0].legend()

        barsDf = histProfDF[:-1]
        ax[1].bar(barsDf.Date - shift, barsDf["Equity"], width, label="Equity")
        ax[1].bar(barsDf.Date - shift, barsDf["Div"], width, bottom=barsDf["Equity"], label="Div")
        ax[1].bar(
            barsDf.Date - shift,
            barsDf[DataSchema.PROFIT],
            width,
            bottom=barsDf["Div"] + barsDf["Equity"],
            label=DataSchema.PROFIT,
        )
        ax[1].bar(barsDf.Date + shift, barsDf["Cost"], width, label="Cost")
        ax[1].legend()
        ax[1].set_ylabel(self.currency.symbol)

        plt.xticks(barsDf["Date"], rotation=90)
        plt.xlabel("Date")
        plt.ylabel("gain")
        plt.title("Historical profitability")
        self.history_chart = fig
        return plt.show()

    def run(self, input: Input = None):
        if input:
            self.input = input
        if self.input == None:
            return

        self.currency = currency_market_map[self.input.broker]
        pd.options.display.float_format = f"{self.currency.symbol} {{:,.2f}}".format
        self.open_dataframe()
        self.load_statement()
        self.load_external_data()
        self.load_recommended_wallet()

        self.merge_statement_data()
        self.merge_external_data()
        self.compute_average_price()
        self.compute_realized_profit()
        self.compute_portfolio()
        self.compute_snapshot()
        self.compute_dividends()
        self.compute_history_snapshot()

    def export_to_excel(self, outfile):
        # Create a Pandas Excel writer using XlsxWriter as the engine.
        writer = pd.ExcelWriter(outfile, engine="xlsxwriter")
        self.snapshot_view.to_excel(writer, sheet_name="snapshot")
        self.portfolio_df.to_excel(writer, sheet_name="portfolio")

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
    config = Input(
        broker=Broker.TDAMERITRADE,
        statement_dir=f"{root}/transactions_td_ameritrade",
        recommended_wallet=f"{root}/transactions_td_ameritrade/global_wallet.json",
    )
    config = Input(
        broker=Broker.CHARLES_SCHWAB,
        statement_dir=f"{root}/transactions_schwab",
        recommended_wallet=f"{root}/transactions_schwab/global_wallet.json",
    )

    wallet = Wallet(root + "/wallet")
    wallet.run(input=config)

    # wallet.export_to_excel(root + 'out.xlsx')
    # wallet.generate_charts()
    # wallet.history_chart.savefig(root + 'chart.png')
    print("Finished")
