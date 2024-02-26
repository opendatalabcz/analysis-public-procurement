import dash
from dash import html, dcc

dash.register_page(__name__,
                   path='/home',
                   title='Home',
                   top_nav=True,
                   name='Home')


def layout():
    layout = html.Div([
        html.H1(
            [
                "Home page"
            ]
        )
    ])
    return layout
