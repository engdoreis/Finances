from dash import Dash, dcc, html
from dash.dependencies import Input, Output

from . import ids


def render(app: Dash, configs) -> html.Div:
    @app.callback(
        Output("output", "children"),
        Input(ids.UPDATE_BTN, "n_clicks"),
    )
    def restart_server(n_clicks):
        if n_clicks is not None:
            for b, w in configs.items():
                w.run()
            return None
        return None

    return html.Div(
        [
            html.Button("Restart Server", id=ids.UPDATE_BTN),
            html.Div(id="output"),
        ]
    )
