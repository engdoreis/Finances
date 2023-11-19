import pandas


class DataSchema:
    DATE = "DATE"
    MONTH = "Month"
    YEAR = "Year"
    SYMBOL = "SYMBOL"
    PRICE = "PRICE"
    QTY = "QUANTITY"
    AMOUNT = "AMOUNT"
    OPERATION = "OPERATION"
    TYPE = "TYPE"
    DESCRIPTION = "DESCRIPTION"
    FEES = "FEE"
    PROFIT = "Profit"
    CASH = "CASH"
    AVERAGE_PRICE = "PM"
    PM_BRL = "PM_BRL"
    PAYDATE = "PAYDATE"
    DAYTRADE = "DayTrade"
    DIV_ACUM = "acumProv"
    QTY_ACUM = "acum_qty"
    DATE_FORMAT = "%Y-%m-%d"

    def base_columns():
        return [
            DataSchema.SYMBOL,
            DataSchema.DATE,
            DataSchema.PRICE,
            DataSchema.QTY,
            DataSchema.OPERATION,
            DataSchema.TYPE,
            DataSchema.FEES,
        ]

    def assert_base_columns(df: pandas.DataFrame):
        for col in DataSchema.base_columns():
            if not col in df.columns:
                raise Exception(f"Column '{col}' not found in table.")
