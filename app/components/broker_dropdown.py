from dash import Dash, dcc, html
from dash.dependencies import Input, Output

import dash_bootstrap_components as dbc
from . import ids


def render(app: Dash, configs) -> html.Div:
    brokers = list(configs.keys())

    # @app.callback(
    #     Output(ids.BROKER_DROPDOWN, "value"),
    #     Input(ids.SELECT_ALL_BROKERS_BTN, "n_clicks"),
    # )
    # def select_all_years(_: int) -> list[str]:
    #     return brokers

    return html.Div(
        children=[
            html.H6("Broker"),
            html.Div(
                children=dbc.Row(
                    [
                        dbc.Col(
                            [
                                dcc.Dropdown(
                                    id=ids.BROKER_DROPDOWN,
                                    options=[{"label": b, "value": b} for b in brokers],
                                    value=brokers[0],
                                    # multi=True,
                                    # placeholder="Select",
                                )
                            ]
                        ),
                        # dbc.Col(
                        #     [
                        #         html.Button(
                        #             className="dropdown-button",
                        #             children=["Select All"],
                        #             id=ids.SELECT_ALL_BROKERS_BTN,
                        #             n_clicks=0,
                        #         )
                        #     ]
                        # ),
                    ]
                )
            ),
        ],
    )
