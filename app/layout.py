from dash import Dash, html, dcc
import dash_bootstrap_components as dbc
from .components import (
    performance_chart,
    asset_chart,
    snapshot_table,
    portfolio_table,
    ids,
    broker_dropdown,
    update_btn,
)


def create_layout(app: Dash, configs) -> html.Div:
    return html.Div(
        className="app-div",
        children=[
            dcc.Location(id="url", refresh=False),
            html.H1(app.title),
            html.Hr(),
            update_btn.render(app, configs),
            html.Div(
                className="dropdown-container",
                children=[
                    broker_dropdown.render(app, configs),
                ],
            ),
            dbc.Row(
                [
                    dbc.Col([snapshot_table.render(app, configs)], width=3),
                    dbc.Col(
                        [
                            portfolio_table.render(app, configs),
                        ],
                        width=8,
                    ),
                ]
            ),
            performance_chart.render(app, configs),
            asset_chart.render(app, configs),
        ],
    )
