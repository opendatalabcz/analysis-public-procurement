import dash
from dash import html, dcc

dash.register_page(__name__,
                   path='/suppliers',
                   title='Suppliers',
                   top_nav=True,
                   name='Suppliers')


def layout():
    layout = html.Div([
        html.H1(
            [
                "Suppliers"
            ]
        )
    ])
    return layout