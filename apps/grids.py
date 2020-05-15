import dash_core_components as dcc
import dash_html_components as html
# from macro import CountryVariables as cv

# class CountryGrids:
    
#     def var_grid(country):
#         return html.Div([
#                     html.Div([
#                         html.H5('Discounted 19 Month Change in 3m Rate', className='country_graph_header'),
#                         dcc.Graph(figure=cv.eighteen_mo_change_in_3m[country], className='country_graph'),
#                     ]),
#                     html.Div([
#                         html.H5('Realized 18m Change in 3m Rate', className='country_graph_header'),
#                         dcc.Graph(figure=cv.realized_18m[country], className='country_graph'),
#                     ]),
#                     html.Div([
#                         html.H5('5y-3m Yield Curve', className='country_graph_header'),
#                         dcc.Graph(figure=cv.yc_across_countries[country], className='country_graph'),
#                     ]),
#                     html.Div([
#                         html.H5('GDP Growth Measurese', className='country_graph_header'),
#                         html.Div([
#                             html.H5('House & CAI Avg (line0)', className='line0'),
#                             html.H5('6m Annualized RGDP (line1)', className='line1'),
#                             html.H5('Potential Real Growth (line2)', className='line2'),
#                         ], className='legend'),
#                         dcc.Graph(figure=cv.gdp_growth_measures[country], className='country_graph'),
#                     ]),
#                     html.Div([
#                         html.H5('Inflation', className='country_graph_header'),
#                         html.Div([
#                             html.H5('Core (line0)', className='line0'),
#                             html.H5('Headline (line1)', className='line1'),
#                         ], className='legend'),
#                         dcc.Graph(figure=cv.inflation_and_core_inflation[country], className='country_graph'),
#                     ]),
#                     html.Div([
#                         html.H5('Current Account (Ann. USD Billions)', className='country_graph_header'),
#                         dcc.Graph(figure=cv.current_account[country], className='country_graph'),
#                     ]),
#                     # html.Div([
#                     #     html.H5('Citi Economic Surprise Index', className='country_graph_header'),
#                     #     dcc.Graph(figure=cv.usa_citi_economic_surprise, className='country_graph'),
#                     # ]),
#                     html.Div([
#                         html.H5('FX Cumulative 90-Day Returns', className='country_graph_header'),
#                         html.Div([
#                             html.H5('Spot Change (line0)', className='line0'),
#                             html.H5('Total Return(line1)', className='line1'),
#                         ], className='legend'),
#                         dcc.Graph(figure=cv.fx_cum_90d_ret[country], className='country_graph'),
#                     ]),
#                     html.Div([
#                         html.H5('Equity Index Cumulative 90-day Returns', className='country_graph_header'),
#                         dcc.Graph(figure=cv.eq_index_cum_90d_ret[country], className='country_graph'),
#                     ]),
#                     html.Div([
#                         html.H5('Rates', className='country_graph_header'),
#                         html.Div([
#                             html.H5('3m (line0)', className='line0'),
#                             html.H5('2y (line1)', className='line1'),
#                             html.H5('5y (line2)', className='line2'),
#                         ], className='legend'),
#                         dcc.Graph(figure=cv.short_rates[country], className='country_graph'),
#                     ]),    
#                 ], className='country_graphs_box')

#     def var_grid_czk_php(country):
#         return html.Div([
#                     html.Div([
#                         html.H5('Discounted 19 Month Change in 3m Rate', className='country_graph_header'),
#                         dcc.Graph(figure=cv.eighteen_mo_change_in_3m[country], className='country_graph'),
#                     ]),
#                     html.Div([
#                         html.H5('Realized 18m Change in 3m Rate', className='country_graph_header'),
#                         dcc.Graph(figure=cv.realized_18m[country], className='country_graph'),
#                     ]),
#                     html.Div([
#                         html.H5('5y-3m Yield Curve', className='country_graph_header'),
#                         dcc.Graph(figure=cv.yc_across_countries[country], className='country_graph'),
#                     ]),
#                     html.Div([
#                         html.H5('GDP Growth Measurese', className='country_graph_header'),
#                         html.Div([
#                             html.H5('House & CAI Avg (line0)', className='line0'),
#                             html.H5('6m Annualized RGDP (line1)', className='line1'),
#                             html.H5('Potential Real Growth (line2)', className='line2'),
#                         ], className='legend'),
#                         dcc.Graph(figure=cv.gdp_growth_measures[country], className='country_graph'),
#                     ]),
#                     html.Div([
#                         html.H5('Inflation', className='country_graph_header'),
#                         html.Div([
#                             html.H5('Core (line0)', className='line0'),
#                             html.H5('Headline (line1)', className='line1'),
#                         ], className='legend'),
#                         dcc.Graph(figure=cv.inflation_and_core_inflation[country], className='country_graph'),
#                     ]),
#                     html.Div([
#                         html.H5('Current Account (Ann. USD Billions)', className='country_graph_header'),
#                         dcc.Graph(figure=cv.current_account_czk_php[country], className='country_graph'),
#                     ]),
#                     # html.Div([
#                     #     html.H5('Citi Economic Surprise Index', className='country_graph_header'),
#                     #     dcc.Graph(figure=cv.usa_citi_economic_surprise, className='country_graph'),
#                     # ]),
#                     html.Div([
#                         html.H5('FX Cumulative 90-Day Returns', className='country_graph_header'),
#                         html.Div([
#                             html.H5('Spot Change (line0)', className='line0'),
#                             html.H5('Total Return(line1)', className='line1'),
#                         ], className='legend'),
#                         dcc.Graph(figure=cv.fx_cum_90d_ret[country], className='country_graph'),
#                     ]),
#                     html.Div([
#                         html.H5('Equity Index Cumulative 90-day Returns', className='country_graph_header'),
#                         dcc.Graph(figure=cv.eq_index_cum_90d_ret[country], className='country_graph'),
#                     ]),
#                     html.Div([
#                         html.H5('Rates', className='country_graph_header'),
#                         html.Div([
#                             html.H5('3m (line0)', className='line0'),
#                             html.H5('2y (line1)', className='line1'),
#                             html.H5('5y (line2)', className='line2'),
#                         ], className='legend'),
#                         dcc.Graph(figure=cv.short_rates[country], className='country_graph'),
#                     ]),    
#                 ], className='country_graphs_box') 