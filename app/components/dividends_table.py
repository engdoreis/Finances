from dash import Dash, html, dash_table
from dash.dependencies import Input, Output
from dash.dash_table import FormatTemplate

from . import ids


def render(app: Dash, configs) -> html.Div:
    @app.callback(
        Output(ids.DIVIDENDS_TABLE, "children"),
        Input(ids.BROKER_DROPDOWN, "value"),
    )
    def update_chart(value):
        df = configs[value].pvt_div_table_raw.reset_index()
        symbol = configs[value].currency.symbol
        money_fmt = FormatTemplate.money(2).symbol_prefix(symbol)
        columns = [
            {"id": df.columns[0], "name": df.columns[0]},
        ]
        columns += [{"id": col, "name": col, "type": "numeric", "format": money_fmt} for col in df.columns[1:]]

        styles = [
            {"if": {"filter_query": "{{{col}}} < 0".format(col=col), "column_id": col}, "color": "red"}
            for col in df.columns[1:]
        ] + [
            {"if": {"filter_query": "{{{col}}} > 0".format(col=col), "column_id": col}, "color": "green"}
            for col in df.columns[1:]
        ]
        print(styles)

        return html.Div(
            [
                html.H6("Dividends"),
                dash_table.DataTable(
                    df.reset_index().to_dict("records"),
                    columns,
                    style_table={
                        "overflowX": "auto",
                        # "maxWidth": "350px",
                        # "maxHeight": "400px",
                    },
                    style_cell={
                        "textAlign": "center",
                        "font_family": "Arial",
                        "font_size": "14px",
                        "height": "12px",
                        "minWidth": "50px",
                        "backgroundColor": "#f9f9f9",  # Background color for cells
                    },
                    style_header={
                        "backgroundColor": "#0074D9",  # Header background color
                        "fontWeight": "bold",
                        "color": "white",  # Header text color
                    },
                    style_data_conditional=styles,
                    style_cell_conditional=[
                        {
                            "if": {"column_id": df.columns[0]},
                            "backgroundColor": "#0074D9",  # Header background color
                            "font_size": "15px",
                            "fontWeight": "bold",
                            "color": "white",  # Header text color
                            "textAlign": "center",  # Align text in the first column to the center
                        }
                    ],
                ),
            ],
            id=ids.DIVIDENDS_TABLE,
        )

    return html.Div(id=ids.DIVIDENDS_TABLE)
