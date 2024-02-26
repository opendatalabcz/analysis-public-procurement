import dash_bootstrap_components as dbc


def create():
    navbar = dbc.NavbarSimple(
        children=[
            dbc.NavItem(dbc.NavLink("Home", href="/home")),
            dbc.NavItem(dbc.NavLink("Suppliers", href="/suppliers")),
            dbc.NavItem(dbc.NavLink("Contractacting authority", href="/contracting_authority")),
            dbc.NavItem(dbc.NavLink("Public procurement", href="/public_procurement"))
        ],
        color="primary",
        fluid=False,
        links_left=True,
        sticky='Top'
    )
    return navbar
