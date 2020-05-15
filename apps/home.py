import dash_core_components as dcc
import dash_html_components as html
from server import app

layout = html.Div([

    html.H1(children='Macro Dashboards',
    style= {
            'textAlign': 'center',
            'padding-top': '30px',
            'color': '#0064ad',
            'font-weight': 'bold'
        }),
    dcc.Link(children=
        html.Button('Sort by Data Type'),
        href='/datatype',
        style={
            'display': 'flex',
            'justify-content': 'center',
            'margin-bottom': '30px',
        }),
    dcc.Link(children=
        html.Button('Choose Country'),
        href='/callback',
        style={
            'display': 'flex',
            'justify-content': 'center',
            'margin-bottom': '30px',
        }),
])