from dash import Dash, html, dash_table
from dash.dependencies import Input, Output
from dash.dash_table import FormatTemplate
from dash.dash_table.Format import Format, Scheme

from data import DataSchema
from . import ids


def render(app: Dash, configs) -> html.Div:
    @app.callback(
        Output(ids.PORTFOLIO_TABLE, "children"),
        Input(ids.BROKER_DROPDOWN, "value"),
    )
    def update_chart(value):
        wallet = configs[value]
        df = wallet.portfolio_df.reset_index()
        symbol = wallet.currency.symbol
        money_fmt = FormatTemplate.money(2).symbol_prefix(symbol)
        pct_fmt = FormatTemplate.percentage(1)
        fixed_fmt = Format(precision=1, scheme=Scheme.fixed)
        columns = [
            dict(id=DataSchema.SYMBOL, name="Ticker", type="text"),
            dict(id=DataSchema.AVERAGE_PRICE, name="Average", type="numeric", format=money_fmt),
            dict(id=DataSchema.PRICE, name="Price", type="numeric", format=money_fmt),
            dict(id=DataSchema.QTY, name="Qty", type="numeric", format=fixed_fmt),
            dict(id="COST", name="Cost", type="numeric", format=money_fmt),
            dict(id="MKT_VALUE", name="Value", type="numeric", format=money_fmt),
            dict(id="DIVIDENDS", name="Dividends", type="numeric", format=money_fmt),
            dict(id=f"GAIN({symbol})", name="Gain", type="numeric", format=money_fmt),
            dict(id="GAIN(%)", name="Gain(%)", type="numeric", format=pct_fmt),
            dict(id=f"GAIN+DIV({symbol})", name="Gain+Div", type="numeric", format=money_fmt),
            dict(id="GAIN+DIV(%)", name="Gain+Div(%)", type="numeric", format=pct_fmt),
            dict(id="ALLOCATION", name="Allocation", type="numeric", format=pct_fmt),
        ]
        if "TARGET" in df.columns:
            columns.append(dict(id="TARGET", name="Target", type="numeric", format=pct_fmt))
        if "TOP_PRICE" in df.columns:
            columns.append(dict(id="TOP_PRICE", name="Limit", type="numeric", format=money_fmt))
        if "PRIORITY" in df.columns:
            columns.append(dict(id="PRIORITY", name="Priority", type="numeric", format=fixed_fmt))
        if "BUY" in df.columns:
            columns.append(dict(id="BUY", name="ToBuy", type="numeric", format=fixed_fmt))

        styles = [
            {"if": {"filter_query": "{{{col}}} < 0".format(col=col), "column_id": col}, "color": "red"}
            for col in df.columns[1:]
        ] + [
            {"if": {"filter_query": "{{{col}}} > 0".format(col=col), "column_id": col}, "color": "green"}
            for col in df.columns[1:]
        ]

        return html.Div(
            [
                html.H6("Positions"),
                dash_table.DataTable(
                    columns=columns,
                    data=df.to_dict("records"),
                    style_table={
                        "overflowX": "auto",
                        "maxWidth": "1200px",
                        "maxHeight": "500px",
                    },
                    style_cell={
                        "textAlign": "center",
                        "font_family": "Arial",
                        "font_size": "14px",
                        "height": "14px",
                        "minWidth": "30px",
                        "backgroundColor": "#f9f9f9",  # Background color for cells
                    },
                    style_header={
                        "backgroundColor": "#0074D9",  # Header background color
                        "font_size": "16px",
                        "fontWeight": "bold",
                        "color": "white",  # Header text color
                    },
                    style_data_conditional=styles,
                    style_cell_conditional=[
                        {
                            "if": {"column_id": DataSchema.SYMBOL},
                            "backgroundColor": "#0074D9",  # Header background color
                            "font_size": "15px",
                            "fontWeight": "bold",
                            "color": "white",  # Header text color
                            "textAlign": "left",  # Align text in the first column to the left
                        }
                    ],
                ),
            ],
            id=ids.PORTFOLIO_TABLE,
        )

    return html.Div(id=ids.PORTFOLIO_TABLE)
