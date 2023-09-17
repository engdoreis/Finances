import Wallet as wl
from dash import Dash
from dash_bootstrap_components.themes import BOOTSTRAP

from app.layout import create_layout

app = Dash(__name__)


def main() -> None:
    root = "/home/doreis/Documents/"
    root += "Investing"

    wallet_t212 = wl.Wallet(root + "/wallet")
    wallet_t212.run(wl.Input(broker=wl.Broker.TRADING212, statement_dir=f"{root}/transactions_trading212"))

    wallet_td = wl.Wallet(root + "/wallet")
    wallet_td.run(
        wl.Input(
            broker=wl.Broker.TDAMERITRADE,
            statement_dir=f"{root}/transactions_td_ameritrade",
            recommended_wallet=f"{root}/transactions_td_ameritrade/global_wallet.json",
        )
    )
    configs = {
        "TDAmeritrade": wallet_td,
        "Trading212": wallet_t212,
    }

    app = Dash(external_stylesheets=[BOOTSTRAP])
    app.title = "Financial Dashboard"
    app.layout = create_layout(app, configs)

    app.run()


if __name__ == "__main__":
    main()
