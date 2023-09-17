from dash import Dash, dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
from Wallet import Wallet
from . import ids


def render(app: Dash, configs) -> html.Div:
    @app.callback(
        Output(ids.PERFORMANCE_CHART, "children"),
        # Input(ids.BROKER_DROPDOWN, 'disabled'),
        Input(ids.BROKER_DROPDOWN, "value"),
    )
    def update_chart(value):
        df = configs[value[0]].historic_profit_df
        line = px.line(df, x="Date", y="profit_growth ibov_growth sp500_growth CDB".split(), width=1500, height=800)
        line.update_xaxes(zeroline=True, zerolinewidth=2, zerolinecolor="Black")
        line.update_yaxes(zeroline=True, zerolinewidth=2, zerolinecolor="Black")
        line.update_yaxes(tickformat=".1%")
        line.update_traces(hovertemplate="<b>%{fullData.name}</b>: %{y}<br>" + "<extra></extra>")

        line.update_layout(hovermode="x unified")

        return html.Div(dcc.Graph(figure=line), id=ids.PERFORMANCE_CHART)

    return html.Div(id=ids.PERFORMANCE_CHART)
