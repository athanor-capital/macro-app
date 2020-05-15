import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
from plotly.tools import mpl_to_plotly
from server import app
import dash_table
import pandas as pd
import athanorm
# from macro import disc_18mo_change_in_3m, realized_18m_graph, yc_across_countries, gdp_growth_measures, inflation_and_core_inflation, current_account, citi_economic_surprise, fx_cum_90d_ret, eq_index_cum_90d_ret, short_rates

df=pd.read_sql("select * from macroind1", athanorm.dev_engine)

layout = html.Div([
    html.H1(children='Sort By Data Type',
    style= {
            'textAlign': 'center',
            'padding-top': '30px',
            'color': '#0064ad',
            'font-weight': 'bold'
        }),
        dash_table.DataTable(
            id='sample-table',
            columns=[{"name": i, "id": i} for i in df.columns],
            data=df.to_dict('records'),
        )
    # dcc.Link(children=
    #     html.Button('Choose Country'),
    #     href='/callback',
    #     style={
    #         'display': 'flex',
    #         'justify-content': 'center',
    #         'margin-bottom': '30px',
    #     }),
    # html.Div([
    #     html.H2('Discounted 18 Month Change in 3m Rate', className='group_header'),
    #     dcc.Graph(id='grid2', figure=disc_18mo_change_in_3m),
    #     html.H2('Realized 18m Change in 3m Rate vs. Currently Discounted 18-month Change in 3m Rate', className='group_header'),
    #     dcc.Graph(id='grid3', figure=realized_18m_graph),
    #     html.H2('5y-3m Yield Curve (bps)', className='group_header'),
    #     dcc.Graph(id='grid4', figure=yc_across_countries),
    #     html.H2('GDP Growth Measures', className='group_header'),
    #     html.Div([
    #         html.H5('House & CAI Avg (line0)', className='line0'),
    #         html.H5('6m Annualized RGDP (line1)', className='line1'),
    #         html.H5('Potential Real Growth (line2)', className='line2'),
    #     ], className='legend'),
    #     dcc.Graph(id='grid5', figure=gdp_growth_measures),
    #     html.H2('Inflation', className='group_header'),
    #     html.Div([
    #         html.H5('Core (line0)', className='line0'),
    #         html.H5('Headline (line1)', className='line1'),
    #     ], className='legend'),
    #     dcc.Graph(id='grid6', figure=inflation_and_core_inflation),
    #     html.H2('Current Account (Ann. USD Billions)', className='group_header'),
    #     dcc.Graph(id='grid7', figure=current_account),
    #     html.H2('Citi Economic Surprise Index', className='group_header'),
    #     dcc.Graph(id='grid8', figure=citi_economic_surprise),
    #     html.H2('FX Cumulative 90-day Returns', className='group_header'),
    #     html.Div([
    #         html.H5('Spot Change (line0)', className='line0'),
    #         html.H5('Total Return(line1)', className='line1'),
    #     ], className='legend'),
    #     dcc.Graph(id='grid9', figure=fx_cum_90d_ret),
    #     html.H2('Equity Index Cumulative 90-day Returns', className='group_header'),
    #     dcc.Graph(id='grid10', figure=eq_index_cum_90d_ret),
    #     html.H2('Rates', className='group_header'),
    #     html.Div([
    #         html.H5('3m (line0)', className='line0'),
    #         html.H5('2y (line1)', className='line1'),
    #         html.H5('5y (line2)', className='line2'),
    #     ], className='legend'),
    #     dcc.Graph(id='grid11', figure=short_rates),
    # ]),

])
