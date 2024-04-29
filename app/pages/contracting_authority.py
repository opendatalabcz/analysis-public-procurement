import dash
import dash_bootstrap_components as dbc
from dash import html, dcc

dash.register_page(__name__,
                   path='/contracting_authority',
                   title='Contracting authority',
                   top_nav=True,
                   name='Contracting authority')


def layout():
    layout = dbc.Container([
        dbc.Row(
            dbc.Col(
                html.Div(html.H3(["Zadavatelé veřejných zakázek"]))
            ),
            align="center"
        ),
        dbc.Row(
            dbc.Col(
                html.Div(html.H6(["Nějaký text"]))
            ),
            align="center"
        ),
        dbc.Row(
            dbc.Col(
                dbc.Input(id="searchbar", placeholder="Zadejte název zadavetele", type="text")
            ),
            align="center"
        )
    ])
    return layout
