import dash
from dash import html, dcc

dash.register_page(__name__,
                   path='/contracting_authority',
                   title='Contracting authority',
                   top_nav=True,
                   name='Contracting authority')


def layout():
    layout = html.Div([
        html.H1(
            [
                "Contracting authority"
            ]
        )
    ])
    return layout