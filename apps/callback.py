import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
# from macro import CountryVariables as cv
# from apps.grids import CountryGrids as cg

from server import app

layout = html.Div([
    html.H1('Enter Country Abbreviation'),
    # dcc.Input(id='country_input', type='text', value='usa', debounce=True),
    # dcc.Link(children=
    #     html.Button('Sort by Data Type'),
    #     href='/datatype',
    #     style={
    #         'display': 'flex',
    #         'justify-content': 'center',
    #         'margin-bottom': '30px',
    #     }),
    # html.Div(id='country_output')
])

# @app.callback(
#     Output(component_id='country_output', component_property='children'),
#     [Input(component_id='country_input', component_property='value')]
# )
# def update_selection_message(input_value):
#     if input_value.lower() == 'czk' or input_value.lower() == 'php':
#         return cg.var_grid_czk_php(input_value.upper())
#     else:
#         return cg.var_grid(input_value.upper())