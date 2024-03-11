import dash_bootstrap_components as dbc


def create():
    navbar = dbc.NavbarSimple(
        children=[
            dbc.Row(
                [dbc.Col(dbc.NavItem(
                    dbc.NavLink("Domů", href="/home",
                                class_name="text-white fs-4"))),
                    dbc.Col(dbc.NavItem(
                        dbc.NavLink("Dodavatelé", href="/suppliers",
                                    class_name="text-white fs-4"))),
                    dbc.Col(dbc.NavItem(
                        dbc.NavLink("Zadavatelé", href="/contracting_authority",
                                    class_name="text-white fs-4"))),
                    dbc.Col(dbc.NavItem(
                        dbc.NavLink("Veřejné zakázky", href="/public_procurement",
                                    class_name="text-white fs-4 text-nowrap")))
                 ],
                class_name="d-flex align-items-center"
            )
        ],
        fluid=True,
        color="primary",
        links_left=True,
        sticky='Top',
    )

    return navbar
