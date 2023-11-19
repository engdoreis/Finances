from dash import Dash, html, dcc, Output, Input
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

TABS = dict()


def home_tab(app: Dash, configs) -> html.Div:
    return html.Div(
        children=[
            dbc.Row(
                [
                    dbc.Col(
                        [
                            snapshot_table.render(app, configs),
                        ],
                        width=3,
                    ),
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


def dividends_tab(app: Dash, configs) -> html.Div:
    return html.Div([html.H3("Tab 2 Content"), html.P("This is the content of Tab 2.")])


def create_layout(app: Dash, configs) -> html.Div:
    # Define callback for updating the tab content
    @app.callback(Output(ids.TAB_CONTENT, "children"), Input(ids.APP_TABS, "value"))
    def render_content(selected_tab):
        return TABS[selected_tab]

    TABS[ids.HOME_TAB] = home_tab(app, configs)
    TABS[ids.DIVIDENDS_TAB] = dividends_tab(app, configs)
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
            # Create the Tabs component
            dcc.Tabs(
                id=ids.APP_TABS,
                value=ids.HOME_TAB,
                children=[
                    # Define individual tabs using Tab components
                    dcc.Tab(label="Performance", value=ids.HOME_TAB),
                    dcc.Tab(label="Dividends", value=ids.DIVIDENDS_TAB),
                ],
            ),
            # Display the content of the selected tab
            html.Div(id=ids.TAB_CONTENT, children=TABS[ids.HOME_TAB]),
        ],
    )
