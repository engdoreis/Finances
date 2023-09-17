from dash import Dash, html, dash_table
from dash.dependencies import Input, Output
from dash.dash_table import FormatTemplate

from . import ids


def render(app: Dash, configs) -> html.Div:
    @app.callback(
        Output(ids.SNAPSHOT_TABLE, "children"),
        Input(ids.BROKER_DROPDOWN, "value"),
    )
    def update_chart(value):
        df = configs[value].performance_snapshot.reset_index()
        columns = [dict(id="Item", name=" ", type="text")]

        if "USD" in df.columns:
            columns.append(dict(id="USD", name="USD", type="numeric", format=FormatTemplate.money(2)))
        if "BRL" in df.columns:
            columns.append(
                dict(id="BRL", name="BRL", type="numeric", format=FormatTemplate.money(2).symbol_prefix("R$"))
            )
        if "GBP" in df.columns:
            columns.append(
                dict(id="GBP", name="GBP", type="numeric", format=FormatTemplate.money(2).symbol_prefix("£"))
            )
        if "%" in df.columns:
            columns.append(dict(id="%", name="Rate", type="numeric", format=FormatTemplate.percentage(1)))
        styles = [
            {"if": {"filter_query": "{{{col}}} < 0".format(col=col), "column_id": col}, "color": "red"}
            for col in df.columns[1:]
        ] + [
            {"if": {"filter_query": "{{{col}}} > 0".format(col=col), "column_id": col}, "color": "green"}
            for col in df.columns[1:]
        ]

        return html.Div(
            [
                html.H6("Performance"),
                dash_table.DataTable(
                    columns=columns,
                    data=df.to_dict("records"),
                    style_table={
                        "overflowX": "auto",
                        "maxWidth": "350px",
                        "maxHeight": "400px",
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
                            "if": {"column_id": "Item"},
                            "backgroundColor": "#0074D9",  # Header background color
                            "fontWeight": "bold",
                            "color": "white",  # Header text color
                            "textAlign": "left",  # Align text in the first column to the left
                        }
                    ],
                ),
            ],
            id=ids.SNAPSHOT_TABLE,
        )

    return html.Div(id=ids.SNAPSHOT_TABLE)
