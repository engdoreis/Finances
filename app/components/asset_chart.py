from dash import Dash, dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objects as go
from . import ids
from data import DataSchema

def render(app: Dash, configs) -> html.Div:
    @app.callback(
        Output(ids.ASSET_CHART, "children"),
        Input(ids.BROKER_DROPDOWN, "value"),
    )
    def update_chart(value):
        df = configs[value].historic_profit_df
        bar = go.Figure(
            data=[
                go.Bar(x=df["Date"], y=df["Equity"], name="Equity", offsetgroup=1),
                go.Bar(x=df["Date"], y=df[DataSchema.PROFIT], name=DataSchema.PROFIT, base=df["Equity"], offsetgroup=1),
                go.Bar(x=df["Date"], y=df["Div"], name="Div", base=df["Equity"] + df[DataSchema.PROFIT], offsetgroup=1),
                go.Bar(x=df.Date, y=df.Cost, name="Cost", offsetgroup=0),
            ]
        )
        bar.update_layout(width=1500, height=350)

        bar.update_traces(hovertemplate="<b>%{fullData.name}</b>: %{y}<br>" + "<extra></extra>")

        bar.update_layout(hovermode="x unified")

        return html.Div(dcc.Graph(figure=bar), id=ids.ASSET_CHART)

    return html.Div(id=ids.ASSET_CHART)
