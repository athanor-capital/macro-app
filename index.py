import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

from apps import datatype, home, callback

from server import server, app


app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content')
])


@app.callback(Output('page-content', 'children'),
              [Input('url', 'pathname')])
def display_page(pathname):
    if pathname == '/':
        return home.layout
    elif pathname == '/datatype':
        return datatype.layout
    elif pathname == '/callback':
        return callback.layout
    else:
        return '404'

if __name__ == '__main__':
    app.debug = True
    from waitress import serve
    serve(app.server)