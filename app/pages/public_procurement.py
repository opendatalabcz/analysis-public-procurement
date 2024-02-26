import dash
from dash import html, dcc

dash.register_page(__name__,
                   path='/public_procurement',
                   title='Public procurement',
                   top_nav=True,
                   name='Public procurement')


def layout():
    layout = html.Div([
        html.H1(
            [
                "Public procurement"
            ]
        )
    ])
    return layout