from dash import Dash, html, dcc
import dash_bootstrap_components as dbc
import dash
from components import navbar

NAVBAR = navbar.create()

app = Dash(__name__, use_pages=True, external_stylesheets=[dbc.themes.FLATLY])

app.layout = dcc.Loading(  # <- Wrap App with Loading Component
    id='loading_page_content',
    children=[
        html.Div(
            [
                dbc.Row(dbc.Col(NAVBAR)),
                dash.page_container

            ]
        )
    ],
    color='primary',  # <- Color of the loading spinner
    fullscreen=True  # <- Loading Spinner should take up full screen
)

if __name__ == '__main__':
    app.run(debug=True, port=5000)