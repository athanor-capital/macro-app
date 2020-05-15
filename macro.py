import numpy as np
import pandas as pd
pd.set_option('display.max_columns', 25)
from IPython.core.display import display, HTML
import math
import athanorm
from athanorm.pivot import pivot_by_cmd
from toolbox.sharpe import get_sharpe_summary, get_sharpe_summary_monthly
from toolbox.chart import plot, multiplot
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
display(HTML("<style>.container { width:100% !important; }</style>"))
from athanorm import pivot
import seaborn as sns

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.dates as mdates
from athanorm import fx
pd.options.display.float_format = '{:,.3f}'.format
pd.options.display.max_columns = None
from toolbox.chart import pretty_plot
from bbg_cult.bbg_followers import Prayer 
bbg = Prayer(host = "http://10.255.232.33:8000")
from toolbox.chart import pretty_multiplot

import matplotlib.ticker as mtick
from matplotlib.backends.backend_pdf import PdfPages


from cycler import cycler

from plotly.tools import mpl_to_plotly
import dash_core_components as dcc


pd.options.display.float_format = '{:,.4f}'.format

print('starting calculations')

def ewma(df,period):
    smooth_df = df.copy()
    smooth_df = smooth_df.ewm(alpha=2./(period+1.)).mean()
    return smooth_df

def multi_plot(df,countries,root,pct,startYear = 1900,zero = False, filename = 'foo2',path='err', grid=False, just_month = False):
    
    startYear = int(startYear)
    #fig = plt.figure(figsize=(18,18))
    fig = plt.figure(figsize=(26,40))
    #gs =gridspec.GridSpec(8, 4)
    #gs =gridspec.GridSpec(12, 4)
    gs =gridspec.GridSpec(12, 4)

    #for i in range(0,len(countries)-1):
    for i in range(0,len(countries)):
        
        if len(df[countries[i]][df[countries[i]]<0]) > 0:
            zero = True
            #plt.autoscale(enable=True)
            #print(countries[i],plt.gca().get_ylim()[0])
            #print(df[countries[i]].min()*2)
            #plt.ylim((df[countries[i]].min()*2,plt.gca().get_ylim()[1]))
        else:
            zero = False
            #plt.ylim((-df[countries[i]].min()*.1,plt.gca().get_ylim()[1])) ####
        
        plt.subplot(gs[i])
        
        
        to_plt = df[countries[i]][df.index.year>=startYear]
        plt.plot(to_plt, color='blue',linewidth=0.75)
        plt.title('\n{}'.format(countries[i]))
        #plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        
        if just_month == True:
            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d-%b'))
            
        else:
            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m-%y'))
        
        if pct == True:
            plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter())

        #plt.ylim((-10,10))
        if zero == True:
            plt.axhline(0,color = 'black')
            
        if grid == True:
            plt.grid(True)
        
       
    fig.suptitle(root,y=1,fontweight="bold")
    plt.tight_layout()
    path.savefig(fig, bbox_inches = 'tight')

    plt.show()
    
    
def multi_series_plot(df1,df2,df3,col_names,title = 'title',labels = ['series1','series2','series3'],startYear = 1900,zero = False, filename = 'foo2',path='err', grid=False, just_month = False):
    startYear = int(startYear)
    fig = plt.figure(figsize=(26,40))
    #gs =gridspec.GridSpec(8, 4)
    gs =gridspec.GridSpec(12, 3, figure = fig)
    
    #for i in range(0,len(countries)-1):
    for i in range(0,len(col_names)):
        plt.subplot(gs[i])
        
        if grid == True:
            plt.grid(True)
        
        if isinstance(df3, pd.DataFrame):
            chart_df = pd.DataFrame({
                'series1':df1['{}'.format(col_names[i])],
                'series2': df2['{}'.format(col_names[i])],
                'series3': df3['{}'.format(col_names[i])]
                
            }
            )#.fillna(method='bfill')

            plt.plot(chart_df[chart_df.index.year >= startYear],linewidth=0.75)
            
            if just_month == True:
                plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d-%b'))

            #cy = cycler('color', ['blue', 'red', 'blue'])
            #ax.set_prop_cycle(cy)
            
            plt.title('\n{}'.format(col_names[i]))
            if i==2:
                plt.legend(labels, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
                
            plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter())


            #plt.ylim((-10,10))
            if zero == True:
                plt.axhline(0,color = 'black')
                
                
        else:
            chart_df = pd.DataFrame({
                'series1':df1['{}'.format(col_names[i])],
                'series2': df2['{}'.format(col_names[i])]
            }
            )#.fillna(method='bfill')
            
            plt.plot(chart_df[chart_df.index.year >= startYear],linewidth=0.75)
            if just_month == True:
                plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d-%b'))
            
            #cycler('color', ['blue', 'red'])
            #ax.set_prop_cycle(cy)

            plt.title('\n{}'.format(col_names[i]))
            if i ==2:
                plt.legend(labels[:2], bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
            plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter())


            #plt.ylim((-10,10))
            if zero == True:
                plt.axhline(0,color = 'black')
    
    fig.suptitle(title,y=1,fontweight="bold")
    plt.tight_layout()
    path.savefig(fig, bbox_inches = 'tight')
    plt.show()
    return
    
    
def pretty_plot(brs, fields=[], figsize=(35,10), title="Title", loc=2):
    if not fields:
        fields = brs.columns.values

    plt.figure(figsize=figsize, dpi=80)

    for cback in fields:

        #plt.plot(brs[cback].dropna(how='all').cumsum(), linewidth='0.7', label=cback)#.replace('_4', ''))infl
        plt.plot(brs[cback], linewidth='0.7', label=cback)#.replace('_4', ''))


    plt.xlabel('Date')
    plt.ylabel('Returns on Book')

    plt.title(title)
    plt.legend(loc=loc)
    #plt.axhline(0, linewidth='1.5', color='black')
    plt.gca().grid(which='major', linestyle='-', linewidth='0.5', color='black')
    plt.gca().grid(which='minor', linestyle=':', linewidth='0.5', color='gray')

    #plt.gca().set_yticklabels(['{:.2f}%'.format(x * 100) for x in plt.gca().get_yticks()]) 
    plt.minorticks_on()

    plt.show()
def ma_chg(df,period, chgType = 'geo', freq = 'd'):
    exponent = (252/((period * 1.5)/2 - 0.5))
    if chgType == 'geo':
        dfChg = (df / df.rolling(int(period*1.5)).mean())**exponent - 1
    else:
        dfChg = (df - df.rolling(int(period*1.5)).mean())*exponent
    
    return dfChg

def dataPull(root,cList, fillna=True):
    if fillna==True:
        df = athanorm.fx.get_values_by_country(root, countries=cList).fillna(method='ffill')
    else:
        df = athanorm.fx.get_values_by_country(root, countries=cList)
    return df

def deriveddataPull(root,cList,ffill=True):
    if ffill == True:
        df = athanorm.fx.get_derived_by_country(root, countries=cList).fillna(method='ffill')
    else:
        df = athanorm.fx.get_derived_by_country(root, countries=cList)
    return df

def em_eur_spot_df():
    countries = ['PLD','CZK','HUN','EUR']
    em_eurFX = dataPull('spotfxveurd',['PLD','CZK','HUN'])
    eurFX = dataPull('spotfxvusdd',['EUR'])
    fx_combo = pd.concat([em_eurFX,eurFX],axis=1)
    
    for i in range(0,len(countries)):
        fx_combo[countries[i]] = fx_combo[countries[i]] * fx_combo['EUR']
    
    return fx_combo[['PLD','CZK','HUN']]
        
def em_eur_fx_ret_df():
    countries = ['PLD','CZK','HUN','EUR']
    em_eurFX = deriveddataPull('3mFwdRetVEURD',['PLD','CZK','HUN'])
    eurFX = deriveddataPull('3mFwdRetVUsdD',['EUR'])
    fx_combo = pd.concat([em_eurFX,eurFX],axis=1)
    
    for i in range(0,len(countries)):
        fx_combo[countries[i]] = fx_combo[countries[i]] + fx_combo['EUR']
    
    return fx_combo[['PLD','CZK','HUN']]
    
def em_eur_fx_carry_df():
    countries = ['PLD','CZK','HUN','EUR']
    em_eurFX = deriveddataPull('3mFwdYldAnnVEURD',['PLD','CZK','HUN'])
    eurFX = deriveddataPull('3mFwdYldAnnVUSDD',['EUR'])
    fx_combo = pd.concat([em_eurFX,eurFX],axis=1)
    
    for i in range(0,len(countries)-1):
        fx_combo[countries[i]] = fx_combo[countries[i]] + fx_combo['EUR']
    
    return fx_combo[['PLD','CZK','HUN']]


def imp_vol_pull():
    imp_vol_df =pd.concat([dataPull('3mFwdVolVUsdD',cList = cCodesAll),dataPull('3mFwdVolVEurD',cList = ['PLD','CZK','HUN'])],axis=1)
    return imp_vol_df

def fx_return(EUR_lst, codes,denom = 'USA'):
    fx_return_df = pd.concat([deriveddataPull('3mFwdRetVUSDD',cList = codes)],axis=1)
    fx_return_df_eurlst = pd.concat([deriveddataPull('3mFwdRetVEURD',cList = EUR_lst)],axis=1)
    #display(fx_return_df_eurlst)
    #display(fx_return_df['EUR'])
    for each in fx_return_df_eurlst:
        fx_return_df_eurlst[each] = (1+fx_return_df_eurlst[each])*(1+fx_return_df['EUR'])-1
    fx_return_df = pd.concat([fx_return_df,fx_return_df_eurlst],axis=1)
    #display(fx_return_df_eurlst)
    if denom == 'USA':
        fx_return_df = fx_return_df
    elif denom == 'EUR':
        eu = fx_return_df[['EUR']].copy()
        fx_return_df = fx_return_df.drop(['EUR'],axis = 1)
        fx_return_df = fx_return_df.subtract(eu.EUR,axis = 'index')
        us = eu[['EUR']] * - 1
        us = us.rename(columns={'EUR': 'USA'})
        fx_return_df = pd.concat([fx_return_df,us],axis=1)
    else:
        jp = fx_return_df[['JPN']].copy()
        fx_return_df = fx_return_df.drop(['JPN'],axis = 1)
        fx_return_df = fx_return_df.subtract(jp.JPN,axis = 'index')
        us = jp[['JPN']] * - 1
        us = us.rename(columns={'JPN': 'USA'})
        fx_return_df = pd.concat([fx_return_df,us],axis=1)
    fx_return_df.index = pd.to_datetime(fx_return_df.index)
    return fx_return_df

    

def spot_df(EUR_lst, codes, denom='USA'):
    
    spotFX_df = pd.concat([dataPull('spotfxvusdd',codes),em_eur_spot_df()],axis = 1)
    spotFX_df_eurlst = pd.concat([dataPull('spotfxveurd',cList = EUR_lst)],axis=1)
    
    for each in EUR_lst:
        spotFX_df_eurlst[each] =  spotFX_df_eurlst[each]*spotFX_df['EUR']
        
    if denom == 'USA':
        spotFX_df = spotFX_df
    
    elif denom == 'EUR':
        eur = spotFX_df[['EUR']].copy()
        spotFX_df = spotFX_df.drop(['EUR'],axis = 1)
        spotFX_df = spotFX_df.divide(eur.EUR,axis = 'index')
        usdeur = 1/eur.copy()
        usdeur = usdeur.rename(columns={'EUR': 'USA'})
        spotFX_df = pd.concat([spotFX_df,usdeur],axis=1)
    
    else:
        jpy = spotFX_df[['JPN']].copy()
        spotFX_df = spotFX_df.drop(['JPN'],axis = 1)
        spotFX_df = spotFX_df.divide(jpy.JPN,axis = 'index')
        usdjpy = 1/jpy.copy()
        usdjpy = usdjpy.rename(columns={'JPN': 'USA'})
        spotFX_df = pd.concat([spotFX_df,usdjpy],axis=1)
    
    return spotFX_df



def equity_fx(denom='USA'):
    equities = dataPull('EqD',cCodesAll)
    spotFX_eq = spot_df(denom=denom)

    if denom == 'USA':    
        spotFX_eq['USA'] = 1
        equities_fx = equities * spotFX_eq
    elif denom == 'EUR':
        spotFX_eq['EUR'] = 1
        equities_fx = equities * spotFX_eq
    else:
        spotFX_eq['JPN'] = 1
        equities_fx = equities * spotFX_eq
        
    return equities_fx


def equity_diff_fx(denom = 'USA'):
    equities_fx = equity_fx(denom=denom)
    
    if denom == 'USA':
        us = equities_fx[['USA']].copy()
        equities_fx = equities_fx.drop(['USA'],axis = 1)
        equities_fx = equities_fx.divide(us.USA,axis = 'index')
    elif denom == 'EUR':
        eu = equities_fx[['EUR']].copy()
        equities_fx = equities_fx.drop(['EUR'],axis = 1)
        equities_fx = equities_fx.divide(eu.EUR,axis = 'index')
    else:
        jp = equities_fx[['JPN']].copy()
        equities_fx = equities_fx.drop(['JPN'],axis = 1)
        equities_fx = equities_fx.divide(jp.JPN,axis = 'index')

    return equities_fx


def equity_diff(denom='USA'):
    equities = dataPull('EqD',cCodesAll)

    if denom == 'USA':
        us = equities[['USA']].copy()
        equities = equities.drop(['USA'],axis = 1)
        equities = equities.divide(us.USA,axis = 'index')
    elif denom == 'EUR':
        eu = equities[['EUR']].copy()
        equities = equities.drop(['EUR'],axis = 1)
        equities = equities.divide(eu.EUR,axis = 'index')
    else:
        jp = equities[['JPN']].copy()
        equities = equities.drop(['JPN'],axis = 1)
        equities = equities.divide(jp.JPN,axis = 'index')

    return equities


def carry(denom='USA'):
    if denom == 'USA':
        carry_data = pd.concat([deriveddataPull('3mFwdYldAnnVUSDD',cList = cCodesAll),em_eur_fx_carry_df()],axis=1)
    elif denom == 'EUR':
        carry_data = pd.concat([deriveddataPull('3mFwdYldAnnVUSDD',cList = cCodesAll),em_eur_fx_carry_df()],axis=1)
        eur = carry_data[['EUR']].copy()
        carry_data = carry_data.drop(['EUR'],axis = 1)
        carry_data = carry_data.subtract(eur.EUR,axis = 'index')
        usaeur = pd.DataFrame(eur * -1)
        usaeur = usaeur.rename(columns={'EUR': 'USA'})
        carry_data = pd.concat([carry_data,usaeur],axis=1)
    else:
        carry_data = pd.concat([deriveddataPull('3mFwdYldAnnVUSDD',cList = cCodesAll),em_eur_fx_carry_df()],axis=1)
        jpy = carry_data[['JPN']].copy()
        carry_data = carry_data.drop(['JPN'],axis = 1)
        carry_data = carry_data.subtract(jpy.JPN,axis = 'index')
        usajpy = jpy * -1
        usajpy = usajpy.rename(columns={'JPN': 'USA'})
        carry_data = pd.concat([carry_data,pd.DataFrame(usajpy)],axis=1)
    return carry_data
        
def inflation_diff(denom = 'USA'):
    inflation_df = inflation()
    
    if denom == 'USA':
        us = inflation_df[['USA']].copy()
        inflation_df = inflation_df.drop(['USA'], axis=1)
        inflation_df = inflation_df.subtract(us.USA,axis='index')
    elif denom == 'EUR':
        eur = inflation_df[['EUR']].copy()
        inflation_df = inflation_df.drop(['EUR'], axis=1)
        inflation_df = inflation_df.subtract(eur.EUR,axis='index')
    else:
        jpn = inflation_df[['JPN']].copy()
        inflation_df = inflation_df.drop(['JPN'], axis=1)
        inflation_df = inflation_df.subtract(jpn.JPN,axis='index')
    return inflation_df


def real_carry(denom='USA'):
    carry_inflation = inflation_diff(denom = denom)
    carry_inflation = carry_inflation.shift(2)
    carry_inflation = (carry_inflation + carry_inflation.rolling(2).mean() + carry_inflation.rolling(3).mean()) / 3.
    carry_data = carry(denom=denom)

    carry_inflation = match_indexes(carry_inflation.resample('D').ffill(),carry_data)

    carry_inflation = carry_inflation.rolling(20).mean()    
    real_carry = carry_data - carry_inflation
    
    return real_carry


def real_fx(denom='USA'):
    spotFX = spot_df(denom=denom)
    
    cpi = dataPull('cpi',cList = cCodesAll).shift(2)
    wpi = dataPull('wpi',cCodesAll).shift(2)
    wpi = wpi.rename(columns = {'Ind':'IND'})
    
    cpi = cpi / cpi.shift(1) - 1
    wpi = wpi / wpi.shift(1) - 1
        
    real_fx_inflation = cpi.combine_first(wpi)
    real_fx_inflation = (1+real_fx_inflation).cumprod()
    
    real_fx_inflation = match_indexes(real_fx_inflation.resample('D').ffill(),spotFX)
    
    if denom == 'USA':
        us = real_fx_inflation[['USA']].copy()
        real_fx_inflation = real_fx_inflation.drop(['USA'], axis=1)
        real_fx_inflation = real_fx_inflation.divide(us.USA,axis='index')
        real_fx_inflation = real_fx_inflation.rolling(20).mean()
    elif denom == 'EUR':
        eu = real_fx_inflation[['EUR']].copy()
        real_fx_inflation = real_fx_inflation.drop(['EUR'], axis=1)
        real_fx_inflation = real_fx_inflation.divide(eu.EUR,axis='index')
        real_fx_inflation = real_fx_inflation.rolling(20).mean()
    else:
        jp = real_fx_inflation[['JPN']].copy()
        real_fx_inflation = real_fx_inflation.drop(['JPN'], axis=1)
        real_fx_inflation = real_fx_inflation.divide(jp.JPN,axis='index')
        real_fx_inflation = real_fx_inflation.rolling(20).mean()
    
    real_fx_df = spotFX * real_fx_inflation
    real_fx_df_mean = real_fx_df.rolling(252*10).mean().fillna(method = 'ffill')
    real_fx_df = real_fx_df / real_fx_df_mean - 1 
    return real_fx_df

def df_rename(df,name):
    df.columns = [name]
    return df

def bdh(code,ffill = True):
    
    try:
        if ffill == True:
            ret = bbg.bdh(code, ['PX_LAST'], pd.to_datetime('1990-01-01').date(), pd.to_datetime('today').date()).fillna(method = 'ffill')
        else:
            ret = bbg.bdh(code, ['PX_LAST'], pd.to_datetime('1990-01-01').date(), pd.to_datetime('today').date())#.fillna(method = 'bfill')
    except:
        ret = pd.DataFrame(pd.Series(np.nan))
    
    return ret

def make_momentum_subframe(names, windows, raw_df_sub, change_or_level, name_lst):
    today = pd.to_datetime('today')
    df = pd.DataFrame()
    
    for each in windows:
        if change_or_level == 'chg':
            df = pd.concat([df,raw_df_sub.diff().rolling(each).sum().loc[today.date(),:]*100],axis=1, sort=False)
        
        else:
            df = pd.concat([df,raw_df_sub.iloc[-each-1,:]*100],axis=1, sort=False)
        
    df.columns = name_lst
    return df

def momentum_barchart(raw_df,windows, names, title, change_or_level,filename =  "foo3.pdf",path='err'):
    
    dm = ['USA','EUR','JPN','GBR','CAN','AUS','NZL','SWE']
    em = ['IND','CHN','TLD','KOR','RUS','CZK','PLD','HUN','TUR','BRZ','MEX','CHI','COL','SAF','IDR','PHP']
    
    em_df = make_momentum_subframe(em, windows, raw_df[em],change_or_level, names)
    dm_df = make_momentum_subframe(dm, windows, raw_df[dm],change_or_level, names)
    
    em_df = em_df.sort_values(em_df.columns[0],ascending = False)
    dm_df = dm_df.sort_values(dm_df.columns[0],ascending = False)
    
    
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(20, 10),sharey='col')
    axes[0].yaxis.set_major_formatter(mtick.PercentFormatter())
    axes[1].yaxis.set_major_formatter(mtick.PercentFormatter())
    em_df.plot.bar(rot=0,ax=axes[0],colormap='Paired')
    dm_df.plot.bar(rot=0,ax=axes[1],colormap='Paired')
    axes[0].set_title('EM')
    axes[1].set_title('DM')
    #fig.subplots_adjust(top=0.88)
    fig.suptitle(title,y=1,fontweight="bold")
    plt.tight_layout()
    path.savefig(fig, bbox_inches = 'tight')
    
    #ax = df.plot.bar(rot=0)
    #ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    #plt.rcParams["figure.figsize"] = [25,10]
    #plt.title(title)
    return

cCodesAll = ['USA','EUR','DEU','FRA','ITA','ESP','JPN','GBR','CAN','AUS','NZL','SWE','NOR','IND','CHN','IDR','TLD','TAI','PHP','KOR','RUS','CZK','PLD','HUN','TUR','BRZ','PER','MEX','CHI','COL','SAF']
cCodesFX = ['EUR','JPN','GBR','CAN','AUS','NZL','SWE','NOR','IND','CHN','IDR','TLD','TAI','PHP','KOR','RUS','CZK','PLD','HUN','TUR','BRZ','PER','MEX','CHI','COL','SAF']
cCodesFXDev= ['EUR','JPN','GBR','CAN','AUS','NZL','SWE','NOR']
cCodesFXEM = ['IND','CHN','IDR','TLD','TAI','PHP','KOR','RUS','CZK','PLD','HUN','TUR','BRZ','PER','MEX','CHI','COL','SAF']
cCodesRates = ['USA','EUR','JPN','GBR','CAN','AUS','NZL','SWE','IND','CHN','TLD','KOR','RUS','CZK','PLD','HUN','TUR','BRZ','MEX','CHI','COL','SAF', 'IDR', 'PHP']
cCodesGrowth = ['USA','EUR','DEU','FRA','ITA','ESP','JPN','GBR','CAN','AUS','NZL','SWE','IND','CHN','TLD','KOR','RUS','CZK','PLD','HUN','TUR','BRZ','MEX','CHI','COL','SAF']
cCodesMajors = ['EUR','JPN','GBR','CAN','AUS','NZL','SWE','NOR','IND','CHN','KOR','RUS','CZK','PLD','HUN','TUR','BRZ','MEX','SAF']
cCodesDev = cCodesFXDev
cCodesDev = cCodesDev + ['USA']
cCodesProduction = ['ARG','AUT','BEL','BRZ','CAN','CHI','CHN','COL','CZK','DEU','ESP','EUR','FRA','GBR','GRC','HUN','IDR','IND','IRE','ITA','JPN','KOR','MAL','MEX','NLD','NOR','NZL','PER','PHP','PLD','PRT','RUS','SAF','SNG','SWE','TAI','TLD','TUR','USA']


cai_codes = [('USA','GSUSCAI Index'),('CHN','GSCNCAI Index'),('EUR','GSEACAI Index'),('CAN','GSCACAI Index'),('IND','GSINCAI Index'),('GBR','GSGBCAI Index'),('BRZ','GSBRCAI Index'),('MEX','GSMXCAI Index'),('AUS','GSAUCAI Index'),('JPN','GSJPCAI Index'),('TUR','GSTRCAI Index'),('CHI','GSCLCAI Index'),('COL','GSCOCAI Index'),('NOR','GSNOCAI Index'),('RUS','GSRUCAI Index'),('CZK','GSCZCAI Index'),('SAF','GSZACAI Index'),('KOR','GSKRCAI Index'),('SWE','GSSECAI Index'),('PLD','GSPLCAI Index'),('ARG','GSARCAI Index'),('TLD','GSTHCAI Index'),('NZL','GSNZCAI Index'),('PER','GSPECAI Index'),('HUN','GSHUCAI Index'),('PHP','GSPHCAI Index'),('TAI','GSTWCAI Index'),('IDR','GSIDCAI Index')]
em_sr_countries = ['IND','CHN','TLD','RUS','CZK','PLD','HUN','TUR','BRZ','MEX','CHI','COL','SAF','IDR','PHP']
dm_sr_countries = [x for x in cCodesRates if x not in em_sr_countries]


fx_daily_rets = fx_return(['HUN','PLD','CZK'],cCodesRates)
fx_daily_rets['USA'] = -fx_daily_rets[['GBR','JPN','EUR']].mean(axis=1)

spot_fx = spot_df(['HUN','PLD','CZK'],cCodesRates)
spot_daily_rets = np.log(spot_fx).diff()
spot_daily_rets['USA'] = -spot_daily_rets[['GBR','JPN','EUR']].mean(axis=1)

equity_prices = pd.DataFrame({'USA': df_rename(bdh('SPX Index'),'USA').iloc[:,0],
                             'EUR': df_rename(bdh('SX5E Index'),'EUR').iloc[:,0],
                             'JPN': df_rename(bdh('NKY Index'),'JPN').iloc[:,0],
                             'GBR': df_rename(bdh('UKX Index'),'GBR').iloc[:,0],
                             'CAN': df_rename(bdh('SPTSX Index'),'CAN').iloc[:,0],
                             'AUS': df_rename(bdh('ASX Index'),'AUS').iloc[:,0],
                             'NZL': df_rename(bdh('NZSE Index'),'NZL').iloc[:,0],
                             'SWE': df_rename(bdh('OMX Index'),'SWE').iloc[:,0],
                             'IND': df_rename(bdh('NIFTY Index'),'IND').iloc[:,0],
                             'CHN': df_rename(bdh('SHCOMP Index'),'CHN').iloc[:,0],
                             'TLD': df_rename(bdh('SET50 Index'),'TLD').iloc[:,0],
                             'KOR': df_rename(bdh('KOSPI Index'),'KOR').iloc[:,0],
                             'RUS': df_rename(bdh('IMOEX Index'),'RUS').iloc[:,0],
                             'CZK': df_rename(bdh('PX Index'),'CZK').iloc[:,0],
                             'PLD': df_rename(bdh('WIG20 Index'),'PLD').iloc[:,0],
                             'HUN': df_rename(bdh('BUX Index'),'HUN').iloc[:,0],
                             'TUR': df_rename(bdh('XU030 Index'),'TUR').iloc[:,0],
                             'BRZ': df_rename(bdh('IBOV Index'),'BRZ').iloc[:,0],
                             'MEX': df_rename(bdh('MEXBOL Index'),'MEX').iloc[:,0],
                             'CHI': df_rename(bdh('IPSA Index'),'CHI').iloc[:,0],
                             'COL': df_rename(bdh('COLCAP Index'),'COL').iloc[:,0],
                             'SAF': df_rename(bdh('TOP40 Index'),'SAF').iloc[:,0],
                             'IDR': df_rename(bdh('JCI Index'),'IDR').iloc[:,0],
                             'PHP': df_rename(bdh('PCOMP Index'),'IDR').iloc[:,0]})

equity_daily_rets = np.log(equity_prices.dropna()).diff()
equity_daily_rets.index = pd.to_datetime(equity_daily_rets.index)

cai_codes = [('USA','GSUSCAI Index'),('CHN','GSCNCAI Index'),('EUR','GSEACAI Index'),('CAN','GSCACAI Index'),('IND','GSINCAI Index'),('GBR','GSGBCAI Index'),('BRZ','GSBRCAI Index'),('MEX','GSMXCAI Index'),('AUS','GSAUCAI Index'),('JPN','GSJPCAI Index'),('TUR','GSTRCAI Index'),('CHI','GSCLCAI Index'),('COL','GSCOCAI Index'),('NOR','GSNOCAI Index'),('RUS','GSRUCAI Index'),('CZK','GSCZCAI Index'),('SAF','GSZACAI Index'),('KOR','GSKRCAI Index'),('SWE','GSSECAI Index'),('PLD','GSPLCAI Index'),('ARG','GSARCAI Index'),('TLD','GSTHCAI Index'),('NZL','GSNZCAI Index'),('PER','GSPECAI Index'),('HUN','GSHUCAI Index'),('PHP','GSPHCAI Index'),('TAI','GSTWCAI Index'),('IDR','GSIDCAI Index')]

USAsr = df_rename(bdh('USSOC Curncy'),'USA')/100
EURsr = df_rename(bdh('EUSWEC Curncy'),'EUR')/100
JPNsr = df_rename(bdh('JYSOC Curncy'),'JPN')/100
GBRsr = df_rename(bdh('BPSWSC Curncy'),'GBR')/100
CANsr = df_rename(bdh('CDSOC Curncy'),'CAN')/100
AUSsr = df_rename(bdh('ADSOC Curncy'),'AUS')/100
NZLsr = df_rename(bdh('NDSOC Curncy'),'NZL')/100
SWEsr = df_rename(bdh('SKSWTNC Curncy'),'SWE')/100
CHNsr = df_rename(bdh('CNRR007 Curncy'),'CHN')/100
INDsr = df_rename(bdh('IN00O/N Curncy'),'IND')/100
#KORsr = df_rename(bdh('KWSWC Curncy'),'KOR')/100
KORsr = df_rename(bdh('KRBO3M Index'),'KOR')/100
TLDsr = df_rename(bdh('THFX1W Index'),'TLD')/100
TURsr = df_rename(bdh('TYUSSWC Curncy'),'TUR')/100
RUSsr = df_rename(bdh('MOSK1W Curncy'),'RUS')/100
PLDsr = df_rename(bdh('WIBO3M Index'),'PLD')/100
#CZKsr = df_rename(bdh('CKSWC Curncy'),'CZK')/100
CZKsr = df_rename(bdh('PRIB03M Index'),'CZK')/100
#HUNsr = df_rename(bdh('HFSWC Curncy'),'HUN')/100
HUNsr = df_rename(bdh('BUBOR03M Index'),'HUN')/100
BRZsr = df_rename(bdh('PREDI30 Index'),'BRZ')/100
MEXsr = df_rename(bdh('MXIBTIIE Curncy'),'MEX')/100
CHIsr = df_rename(bdh('CHOVCHOV Curncy'),'CHI')/100
COLsr = (df_rename(bdh('CORRRMIN Curncy'),'COL')/100).combine_first(df_rename(bdh('CORRRMIN Index'),'COL')/100)
SAFsr = df_rename(bdh('JIBA1M Curncy'),'SAF')/100
IDRsr = df_rename(bdh('IHSWOOC Curncy'),'IDR')/100
PHPsr = df_rename(bdh('PPSWOC Curncy'),'PHP')/100

USAone = df_rename(bdh('USSO1 Curncy'),'USA')/100
EURone = df_rename(bdh('EUSWE1 Curncy'),'EUR')/100
JPNone = df_rename(bdh('JYSO1 Curncy'),'JPN')/100
GBRone = df_rename(bdh('BPSWS1 Curncy'),'GBR')/100
CANone = df_rename(bdh('CDSO1 Curncy'),'CAN')/100
AUSone = df_rename(bdh('ADSO1 Curncy'),'AUS')/100
NZLone = df_rename(bdh('NDSO1 Curncy'),'NZL')/100
SWEone = df_rename(bdh('SKSWTN1 Curncy'),'SWE')/100
CHNone = df_rename(bdh('CGUSSW1 Curncy'),'CHN')/100
INDone = df_rename(bdh('IRSWNI1 Curncy'),'IND')/100
KORone = df_rename(bdh('KWSWO1 Curncy'),'KOR')/100
TLDone = df_rename(bdh('TBUSSW1 Curncy'),'TLD')/100
TURone = df_rename(bdh('TYUSSW1 Curncy'),'TUR')/100
RUSone = df_rename(bdh('RRSWM1 Curncy'),'RUS')/100
PLDone = df_rename(bdh('PZSW1 Curncy'),'PLD')/100
CZKone = df_rename(bdh('CKSW1 Curncy'),'CZK')/100
HUNone = df_rename(bdh('HFSW1 Curncy'),'HUN')/100
BRZone = df_rename(bdh('PREDI360 Index'),'BRZ')/100
MEXone = df_rename(bdh('MPSW1A Curncy'),'MEX')/100
CHIone = df_rename(bdh('CHSWP1 Curncy'),'CHI')/100
COLone = df_rename(bdh('CLSWIB1 Curncy'),'COL')/100
SAFone = df_rename(bdh('SASW1 Curncy'),'SAF')/100
IDRone = df_rename(bdh('IHSWOO1 Curncy'),'IDR')/100
PHPone = df_rename(bdh('PPSWO1 Curncy'),'PHP')/100

USAtwo = df_rename(bdh('USSO2 Curncy'),'USA')/100
EURtwo = df_rename(bdh('EUSWE2 Curncy'),'EUR')/100
JPNtwo = df_rename(bdh('JYSO2 Curncy'),'JPN')/100
GBRtwo = df_rename(bdh('BPSWS2 Curncy'),'GBR')/100
CANtwo = df_rename(bdh('CDSO2 Curncy'),'CAN')/100
AUStwo = df_rename(bdh('ADSO2 Curncy'),'AUS')/100
NZLtwo = df_rename(bdh('NDSO2 Curncy'),'NZL')/100
SWEtwo = df_rename(bdh('SKSWTN2 Curncy'),'SWE')/100
CHNtwo = df_rename(bdh('CGUSSW2 Curncy'),'CHN')/100
INDtwo = df_rename(bdh('IRSWNI2 Curncy'),'IND')/100
KORtwo = df_rename(bdh('KWSWO2 Curncy'),'KOR')/100
TLDtwo = df_rename(bdh('TBUSSW2 Curncy'),'TLD')/100
TURtwo = df_rename(bdh('TYUSSW2 Curncy'),'TUR')/100
RUStwo = df_rename(bdh('RRSWM2 Curncy'),'RUS')/100
PLDtwo = df_rename(bdh('PZSW2 Curncy'),'PLD')/100
CZKtwo = df_rename(bdh('CKSW2 Curncy'),'CZK')/100
HUNtwo = df_rename(bdh('HFSW2 Curncy'),'HUN')/100
BRZtwo = df_rename(bdh('PREDI720 Index'),'BRZ')/100
MEXtwo = df_rename(bdh('MPSW2B Curncy'),'MEX')/100
CHItwo = df_rename(bdh('CHSWP2 Curncy'),'CHI')/100
COLtwo = df_rename(bdh('CLSWIB2 Curncy'),'COL')/100
SAFtwo = df_rename(bdh('SASW2 Curncy'),'SAF')/100
IDRtwo = df_rename(bdh('IHSWOO2 Curncy'),'IDR')/100
PHPtwo = df_rename(bdh('PPSWO2 Curncy'),'PHP')/100

USAfive = df_rename(bdh('USSO5 Curncy'),'USA')/100
EURfive = df_rename(bdh('EUSWE5 Curncy'),'EUR')/100
JPNfive = df_rename(bdh('JYSO5 Curncy'),'JPN')/100
GBRfive = df_rename(bdh('BPSWS5 Curncy'),'GBR')/100
CANfive = df_rename(bdh('CDSO5 Curncy'),'CAN')/100
AUSfive = df_rename(bdh('ADSO5 Curncy'),'AUS')/100
NZLfive = df_rename(bdh('NDSO5 Curncy'),'NZL')/100
SWEfive = df_rename(bdh('SKSWTN5 Curncy'),'SWE')/100
CHNfive = df_rename(bdh('CGUSSW5 Curncy'),'CHN')/100
INDfive = df_rename(bdh('IRSWNI5 Curncy'),'IND')/100
KORfive = df_rename(bdh('KWSWO5 Curncy'),'KOR')/100
TLDfive = df_rename(bdh('TBUSSW5 Curncy'),'TLD')/100
TURfive = df_rename(bdh('TYUSSW5 Curncy'),'TUR')/100
RUSfive = df_rename(bdh('RRSWM5 Curncy'),'RUS')/100
PLDfive = df_rename(bdh('PZSW5 Curncy'),'PLD')/100
CZKfive = df_rename(bdh('CKSW5 Curncy'),'CZK')/100
HUNfive = df_rename(bdh('HFSW5 Curncy'),'HUN')/100
BRZfive = df_rename(bdh('PRDI1800 Index'),'BRZ')/100
MEXfive = df_rename(bdh('MPSW5B Curncy'),'MEX')/100
CHIfive = df_rename(bdh('CHSWP5 Curncy'),'CHI')/100
COLfive = df_rename(bdh('CLSWIB5 Curncy'),'COL')/100
SAFfive = df_rename(bdh('SASW5 Curncy'),'SAF')/100
IDRfive = df_rename(bdh('IHSWOO5 Curncy'),'IDR')/100
PHPfive = df_rename(bdh('PPSWO5 Curncy'),'PHP')/100

shortRate = pd.DataFrame({'USA':USAsr.USA,
                         'EUR':EURsr.EUR,
                         'JPN':JPNsr.JPN,
                         'GBR':GBRsr.GBR,
                         'CAN':CANsr.CAN,
                         'AUS':AUSsr.AUS,
                         'NZL':NZLsr.NZL,
                         'SWE':SWEsr.SWE,
                         'CHN':CHNsr.CHN,
                         'IND':INDsr.IND,
                         'KOR':KORsr.KOR,
                         'TLD':TLDsr.TLD,
                         'TUR':TURsr.TUR,
                         'RUS':RUSsr.RUS,
                         'PLD':PLDsr.PLD,
                         'CZK':CZKsr.CZK,
                         'HUN':HUNsr.HUN,
                         'BRZ':BRZsr.BRZ,
                         'MEX':MEXsr.MEX,
                         'CHI':CHIsr.CHI,
                         'COL':COLsr.COL,
                         'SAF':SAFsr.SAF,
                         'IDR':IDRsr.IDR,
                         'PHP':PHPsr.PHP})

oneRate = pd.DataFrame({'USA':USAone.USA,
                         'EUR':EURone.EUR, 
                         'JPN':JPNone.JPN, 
                         'GBR':GBRone.GBR, 
                         'CAN':CANone.CAN,
                         'AUS':AUSone.AUS, 
                         'NZL':NZLone.NZL, 
                         'SWE':SWEone.SWE, 
                         'CHN':CHNone.CHN,
                         'IND':INDone.IND,
                         'KOR':KORone.KOR,
                         'TLD':TLDone.TLD,
                         'TUR':TURone.TUR,
                         'RUS':RUSone.RUS,
                         'PLD':PLDone.PLD,
                         'CZK':CZKone.CZK,
                         'HUN':HUNone.HUN,
                         'BRZ':BRZone.BRZ,
                         'MEX':MEXone.MEX,
                         'CHI':CHIone.CHI,
                         'COL':COLone.COL,
                         'SAF':SAFone.SAF,
                         'IDR':IDRone.IDR,
                         'PHP':PHPone.PHP})

twoRate = pd.DataFrame({'USA':USAtwo.USA,
                         'EUR':EURtwo.EUR,
                         'JPN':JPNtwo.JPN,
                         'GBR':GBRtwo.GBR,
                         'CAN':CANtwo.CAN,
                         'AUS':AUStwo.AUS,
                         'NZL':NZLtwo.NZL,
                         'SWE':SWEtwo.SWE,
                         'CHN':CHNtwo.CHN,
                         'IND':INDtwo.IND,
                         'KOR':KORtwo.KOR,
                         'TLD':TLDtwo.TLD,
                         'TUR':TURtwo.TUR,
                         'RUS':RUStwo.RUS,
                         'PLD':PLDtwo.PLD,
                         'CZK':CZKtwo.CZK,
                         'HUN':HUNtwo.HUN,
                         'BRZ':BRZtwo.BRZ,
                         'MEX':MEXtwo.MEX,
                         'CHI':CHItwo.CHI,
                         'COL':COLtwo.COL,
                         'SAF':SAFtwo.SAF,
                         'IDR':IDRtwo.IDR,
                         'PHP':PHPtwo.PHP})

fiveRate = pd.DataFrame({'USA':USAfive.USA,
                          'EUR':EURfive.EUR,
                          'JPN':JPNfive.JPN,
                          'GBR':GBRfive.GBR,
                          'CAN':CANfive.CAN,
                          'AUS':AUSfive.AUS,
                          'NZL':NZLfive.NZL,
                          'SWE':SWEfive.SWE,
                          'CHN':CHNfive.CHN,
                          'IND':INDfive.IND,
                          'KOR':KORfive.KOR,
                          'TLD':TLDfive.TLD,
                          'TUR':TURfive.TUR,
                          'RUS':RUSfive.RUS,
                          'PLD':PLDfive.PLD,
                          'CZK':CZKfive.CZK,
                          'HUN':HUNfive.HUN,
                          'BRZ':BRZfive.BRZ,
                          'MEX':MEXfive.MEX,
                          'CHI':CHIfive.CHI,
                          'COL':COLfive.COL,
                          'SAF':SAFfive.SAF,
                          'IDR':IDRfive.IDR,
                          'PHP':PHPfive.PHP})

one_one = (1+twoRate)**2 / (1+oneRate) - 1
disc_tight = (one_one - shortRate).fillna(method='ffill')
yc = fiveRate - shortRate


# Discounted Tightening Today vs. Realized SR
disc_tight_spot = disc_tight.copy()

for country in disc_tight_spot.columns.values:
    disc_tight_spot[country] = disc_tight_spot[country][-1]
    
#multi_series_plot((shortRate - shortRate.shift(378))['2000':]*100,disc_tight_spot['2000':]*100,0,cCodesRates,title = 'Realized 18-month Changes in 3m Rate vs. Currently Discounted 18-month Change in 3m Rate',labels = ['Realized','Disc. Change',0],startYear = 2000,zero = True, filename = 'Realized Change vs Discounted Change',path = pdf)

growth_inhouse = deriveddataPull('CoinGrwth6m',cList = cCodesAll,ffill = False)
growth_goldman = pd.DataFrame()
for each in cai_codes:
    pull = df_rename(bdh(each[1],ffill=False),each[0]).resample('1M').mean()
    if abs(pull.loc[pull.index[-1],each[0]]-pull.loc[pull.index[-2],each[0]])<.0001:
        pull.loc[pull.index[-1],each[0]] = np.nan
    #display(pull)
    #pull = df_rename(pull,each[0])
    growth_goldman = pd.concat([growth_goldman, pull.resample('1M').mean()/100],axis=1)

growth_goldman = (growth_goldman+growth_goldman.shift(3))/2

#
rgdp =  dataPull('rgdp',cList = cCodesAll, fillna = False)
rgdp = rgdp.rename(columns = {'Usa':'USA'})#
rgdp_growth = rgdp.pct_change(periods=6,fill_method='bfill')*2


#
rpot_growth = deriveddataPull('RealPotGR',cList = cCodesAll, ffill = False)
#
gs_house_average = (growth_inhouse+growth_goldman)/2
gs_house_extend = gs_house_average.combine_first(growth_inhouse)

# Add EM, DM, global Averages
gs_house_extend['EM'] = gs_house_extend[em_sr_countries].mean(axis = 1, skipna = False) 
gs_house_extend['DM'] = gs_house_extend[dm_sr_countries].mean(axis=1, skipna = False)
gs_house_extend['Global'] = gs_house_extend[['EM','DM']].mean(axis=1, skipna = False) 


rgdp_growth['EM'] = (rgdp[em_sr_countries].pct_change(periods=6,fill_method='ffill')*2).mean(axis = 1, skipna = False)
rgdp_growth['DM'] = (rgdp[dm_sr_countries].pct_change(periods=6,fill_method='ffill')*2).mean(axis = 1, skipna = False) 
rgdp_growth['Global'] = (rgdp_growth[['EM','DM']].mean(axis = 1, skipna = False)) 


rpot_growth['EM'] = rpot_growth[em_sr_countries].fillna(method='ffill').mean(axis = 1, skipna = False) 
rpot_growth['DM'] = rpot_growth[dm_sr_countries].fillna(method='ffill').mean(axis = 1, skipna = False)
rpot_growth['Global'] = rpot_growth[['EM','DM']].fillna(method='ffill').mean(axis = 1, skipna = False) 

cpi = dataPull('cpi',cList = cCodesAll,fillna = False)
cpicore = dataPull('cpicore',cList = cCodesAll,fillna = False)

inflation =( cpi / cpi.rolling(18).mean())**(12/8.5) - 1
inflation['AUS'] = cpi['NZL'] / cpi['NZL'].shift(12)-1
inflation['NZL'] = cpi['NZL'] / cpi['NZL'].shift(12)-1
inflation['CHI'][:'2011']=np.nan

inflation['EM'] = inflation[em_sr_countries].fillna(method='ffill').mean(axis = 1, skipna = False)
inflation['DM'] = inflation[dm_sr_countries].fillna(method='ffill').mean(axis = 1, skipna = False)
inflation['Global'] = inflation[['EM','DM']].fillna(method='ffill').mean(axis = 1, skipna = False)

inflation_core =( cpicore / cpicore.rolling(18).mean())**(12/8.5) - 1
inflation_core['AUS'] = cpicore['AUS'] / cpicore['AUS'].shift(12)-1
inflation_core['NZL'] = cpicore['NZL'] / cpicore['NZL'].shift(12)-1
inflation_core['COL'] = np.nan
inflation_core['EM'] = inflation_core[em_sr_countries].drop('COL',axis=1).fillna(method='ffill').mean(axis = 1, skipna = False) #dropping COL from EM
inflation_core['DM'] = inflation_core[dm_sr_countries].fillna(method='ffill').mean(axis = 1, skipna = False)
inflation_core['Global'] = inflation_core[['EM','DM']].fillna(method='ffill').mean(axis = 1, skipna = False)



inflation = inflation*100
inflation_core = inflation_core*100 


macro_ind = pd.DataFrame({'DM Discounted Change in 3m Rate':df_rename(pd.DataFrame(disc_tight.fillna(method='ffill')[['USA','EUR','JPN']].mean(axis=1)),'disc_tight').disc_tight*100,
                          'EM Discounted Change in 3m Rate':df_rename(pd.DataFrame(disc_tight.fillna(method='ffill')[em_sr_countries].median(axis=1)),'em_disc_tight').em_disc_tight*100,
                         'DM Mean 3m Rate':df_rename(pd.DataFrame(shortRate.fillna(method='ffill')[['USA','EUR','JPN']].mean(axis=1)),'dev_sr').dev_sr*100,
                          'EM Mean 3m Rate':df_rename(pd.DataFrame(shortRate.fillna(method='ffill')[em_sr_countries].median(axis=1)),'em_sr').em_sr*100,
                          'US 10yr Nominal Rate':df_rename(bdh('USGG10YR Index'),'us_lr_nom').us_lr_nom,
                         'US 10yr Real Rate':df_rename(bdh('GTII10 Govt'),'us_lr_real').us_lr_real,
                          'US 10yr BEI':df_rename(bdh('USGGBE10 Index'),'us_lr_bei').us_lr_bei,
                         'US 10yr-2yr Yield Curve':df_rename(bdh('USGG10YR Index') - bdh('GT02 Govt'),'yc').yc/100,
                         'EMBI Spread':df_rename(bdh('JPEIGLSP Index'),'embi_spread').embi_spread/100,
                         'Corp Bond Spread':df_rename(bdh('LUACOAS Index'),'corp_spread').corp_spread,
                         'HY Spread':df_rename(bdh('LF98OAS Index'),'hy_spread').hy_spread,
                         'EM Carry':df_rename(bdh('FXCTEM8 Index'),'em_carry').em_carry/100})
                          
                          
macro_ind2 = pd.DataFrame({'DXY/100':df_rename(bdh('DXY Curncy'),'dxy').dxy/100,
                         'VIX/100':df_rename(bdh('vix Index'),'vix').vix/100,
                         'EEM/100':df_rename(bdh('MXEF Index'),'eem').eem/100,
                         'SPX/100':df_rename(bdh('spx Index'),'spx').spx/100})
#multi_series_plot(inflation_core['1997':],inflation['1997':],0,cCodesRates, title = 'Inflation',labels = ['Core','Headline',0],startYear = 1997,zero = True,path = pdf)

# Current Account
# Returns dataframe of current accounts for each country, measured in annualized USD Billions
def get_ca_usd_ann(cList, to_smoothe):
    ret = pd.DataFrame()
    for each in cList:
        
        # Get relevant metadata
        metadata = pd.read_sql("select * from athanor.metadata where code_name = '{}CurrAcct'".format(each), athanorm.athanor_engine)
        ann_factor = metadata['ann_factor'].values[0]
        currency = metadata['currency'].values[0]
        unit = metadata['unit'].values[0]
        
        # Get current account
        ca = df_rename(pd.read_sql("select * from athanor.values_data where code_name = '{}CurrAcct'".format(each), athanorm.athanor_engine,index_col = 'datadate').drop('code_name',axis=1),each)
        ca.index= pd.to_datetime(ca.index)
        # Annualize
        ca = ca*ann_factor
        # Turn to billions of FX
        ca = ca*(unit/10**9)
        #print(ca)
        
        # Decide if updated quarterly or monthly
        ## Assumes at least small change with each pub ##
        if ca.iloc[-1,:].values[0]==ca.iloc[-2,:].values[0]:
            quarterly = True
            # Resample Quarterly if needed:
            ca = ca.resample('Q').mean()
            #print("quarterly")
        else:
            # else monthly
            quarterly = False
            
        fx = np.nan
        # Get currency if needed
        if currency == 'Local' and each != 'USA':
            # Dollars per FX
            fx = df_rename(pd.read_sql("select * from athanor.values_data where code_name = '{}spotfxvusdd'".format(each), athanorm.athanor_engine,index_col = 'datadate').drop('code_name',axis=1),each)
            
            # If an EU Country, use EUR fx
            if each in {'DEU','FRA','ITA','ESP'}:
                fx = df_rename(pd.read_sql("select * from athanor.values_data where code_name = 'EURspotfxvusdd'", athanorm.athanor_engine,index_col = 'datadate').drop('code_name',axis=1),each)
            
            # Hungary and CZK need to go to euro then dollar
            if each in {'HUN','CZK'}:
                # eur per fx
                fx = df_rename(pd.read_sql("select * from athanor.values_data where code_name = '{}spotfxveurd'".format(each), athanorm.athanor_engine,index_col = 'datadate').drop('code_name',axis=1),each)
                # usd per eur
                eurusd = df_rename(pd.read_sql("select * from athanor.values_data where code_name = 'eurspotfxvusdd'", athanorm.athanor_engine,index_col = 'datadate').drop('code_name',axis=1),each)
                fx = eurusd*fx
                
            fx.index= pd.to_datetime(fx.index)
            
            # resample FX to monthly or quarterly
            if quarterly ==True:
                fx = fx.resample('Q').mean()
            else:
                fx = fx.resample('1M').mean()
            
            # Convert to dollars using like-timescales
            ca = ca*fx
        ca = ca.resample('1M').ffill()
            
        # Resample again to monthly
        ret = pd.concat([ret,pd.DataFrame(ca)],axis = 1,sort=False)
        #display(ret)
    for each in to_smoothe:
        ret[each] = ret[each].rolling(3).mean()
        #display(ret)
    return ret

swap_codes = pd.read_csv('swap_codes.csv',index_col = 'Tenor')
#swap_codes.columns[:-2]
#print(swap_codes.columns)
# for each column load the swap rates at all tenors

sr = pd.concat([df_rename(bdh(swap_codes.loc['3m',country]+" Index"),country) for country in swap_codes.columns], axis =1)
sr = sr.resample('B').mean()

one = pd.concat([df_rename(bdh(swap_codes.loc['1yr',country]+" Index"),country) for country in swap_codes.columns], axis =1)
two = pd.concat([df_rename(bdh(swap_codes.loc['2yr',country]+" Index"),country) for country in swap_codes.columns], axis =1)
three = pd.concat([df_rename(bdh(swap_codes.loc['3yr',country]+" Index"),country) for country in swap_codes.columns], axis =1)
four = pd.concat([df_rename(bdh(swap_codes.loc['4yr',country]+" Index"),country) for country in swap_codes.columns], axis =1)
five = pd.concat([df_rename(bdh(swap_codes.loc['5yr',country]+" Index"),country) for country in swap_codes.columns], axis =1)
six = pd.concat([df_rename(bdh(swap_codes.loc['6yr',country]+" Index"),country) for country in swap_codes.columns], axis =1)
seven = pd.concat([df_rename(bdh(swap_codes.loc['7yr',country]+" Index"),country) for country in swap_codes.columns], axis =1)
eight = pd.concat([df_rename(bdh(swap_codes.loc['8yr',country]+" Index"),country) for country in swap_codes.columns], axis =1)
nine = pd.concat([df_rename(bdh(swap_codes.loc['9yr',country]+" Index"),country) for country in swap_codes.columns], axis =1)
ten =pd.concat([df_rename(bdh(swap_codes.loc['10yr',country]+" Index"),country) for country in swap_codes.columns], axis =1)

# Interpolate spot curves to construct forward curves over all dates
from datetime import date, timedelta
today = pd.to_datetime('today').date()
yesterday = today - timedelta(days=1)

day = today

# spot curve
spot_yc = pd.DataFrame(index = [.25,1, 2, 3, 4, 5, 6, 7, 8, 9, 10], columns = swap_codes.columns)
for each in spot_yc:
    spot_yc[each] = [eval(i).loc[day,each] for i in ['sr','one','two','three','four','five','six','seven','eight','nine','ten']]
spot_interp = spot_yc.interpolate()

for i,each in enumerate(['sr','one','two','three','four','five','six','seven','eight','nine','ten']):
    eval(each).loc[day,:] = [i for i in spot_interp.iloc[i,:]]

# wrd year rate in number years
one0 = one.copy()
one1 = (((1+two/100)**2)/(1+one/100) - 1)*100
one2 = (((1+three/100)**3)/((1+two/100)**2) - 1)*100
one3 = (((1+four/100)**4)/((1+three/100)**3) - 1)*100
one4 = (((1+five/100)**5)/((1+four/100)**4) - 1)*100
one5 = (((1+six/100)**6)/((1+five/100)**5) - 1)*100
one6 = (((1+seven/100)**7)/((1+six/100)**6) - 1)*100
one7 = (((1+eight/100)**8)/((1+seven/100)**7) - 1)*100
one8 = (((1+nine/100)**9)/((1+eight/100)**8) - 1)*100
one9 = (((1+ten/100)**10)/((1+nine/100)**9) - 1)*100
        

# Construct forward_curves

fwd_yc = pd.DataFrame(index = [float(i) for i in [1,2,3,4,5,6,7,8,9,10]], columns = swap_codes.columns)
for each in fwd_yc.columns:
    fwd_yc[each] = [eval('one{}'.format(i)).loc[day,each] for i in range(10)]

def multi_series_plot_nontime(df1,df2,col_names,title = 'title',labels = ['series1','series2'],zero = False, filename = 'foo2',path='err'):
    fig = plt.figure(figsize=(26,40))
    #gs =gridspec.GridSpec(8, 4)
    gs =gridspec.GridSpec(12, 3, figure = fig)
    
    #for i in range(0,len(countries)-1):
    for i in range(0,len(col_names)):
        zero=False
        plt.subplot(gs[i])
        plt.grid(True)


        chart_df = pd.DataFrame({
            'series1':df1['{}'.format(col_names[i])],
            'series2': df2['{}'.format(col_names[i])]
        }
        )#.fillna(method='bfill')
        
        if len(chart_df[chart_df<0].dropna(how='all')) > 0:
            #display(chart_df[chart_df<0])
            zero = True
        else:
            zero = False
            
        if zero == True:
            plt.axhline(0,color = 'black')
        
        #print(col_names[i])
        display(chart_df)

        plt.plot(chart_df['series1'][np.isfinite(chart_df['series1'])],linewidth=0.75, linestyle='-', marker='o')
        plt.plot(chart_df['series2'][np.isfinite(chart_df['series2'])],linewidth=0.75, linestyle='-')

        #cycler('color', ['blue', 'red'])
        #ax.set_prop_cycle(cy)

        plt.title('\n{}'.format(col_names[i]))
        if i ==2:
            plt.legend(labels[:2], bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter())


        #plt.ylim((-10,10))
            
    
    fig.suptitle(title,y=1,fontweight="bold")
    plt.tight_layout()
    path.savefig(fig, bbox_inches = 'tight')
    plt.show()
    return

bdh('CESICNY Index',ffill = False)['PX_LAST']

# Surprise Index
citi_surprise = pd.DataFrame({'USA':bdh('CESIUSD Index',ffill = False)['PX_LAST'],
                         'EUR':bdh('CESIEUR Index',ffill = False)['PX_LAST'],
                         'JPN':bdh('CESIJPY Index',ffill = False)['PX_LAST'],
                         'GBR':bdh('CESIGBP Index',ffill = False)['PX_LAST'],
                         'CAN':bdh('CESICAD Index',ffill = False)['PX_LAST'],
                         'AUS':bdh('CESIAUD Index',ffill = False)['PX_LAST'],
                         'NZL':bdh('CESINZD Index',ffill = False)['PX_LAST'],
                         'SWE':bdh('CESISEK Index',ffill = False)['PX_LAST'],
                         'CHN':bdh('CESICNY Index',ffill = False)['PX_LAST'],
                         'EM':bdh('CESIEM Index',ffill = False)['PX_LAST'],
                         'DM':bdh('CESIG10 Index',ffill = False)['PX_LAST'],
                         'World':bdh('CESIGL Index',ffill = False)['PX_LAST']})


def img_multi_plot(df,countries,root,pct,startYear = 1900,zero = False, filename = 'foo2',path='err', grid=False, just_month = False):
    
    startYear = int(startYear)
    #fig = plt.figure(figsize=(18,18))
    fig = plt.figure(figsize=(26,40))
    #gs =gridspec.GridSpec(8, 4)
    #gs =gridspec.GridSpec(12, 4)
    gs =gridspec.GridSpec(12, 4)

    #for i in range(0,len(countries)-1):
    for i in range(0,len(countries)):
        
        if len(df[countries[i]][df[countries[i]]<0]) > 0:
            zero = True
            #plt.autoscale(enable=True)
            #print(countries[i],plt.gca().get_ylim()[0])
            #print(df[countries[i]].min()*2)
            #plt.ylim((df[countries[i]].min()*2,plt.gca().get_ylim()[1]))
        else:
            zero = False
            #plt.ylim((-df[countries[i]].min()*.1,plt.gca().get_ylim()[1])) ####
        
        plt.subplot(gs[i])
        
        
        to_plt = df[countries[i]][df.index.year>=startYear]
        plt.plot(to_plt, color='blue',linewidth=0.75)
        plt.title('\n{}'.format(countries[i]))
        #plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        
        if just_month == True:
            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d-%b'))
            
        else:
            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m-%y'))
        
        if pct == True:
            plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter())

        #plt.ylim((-10,10))
        if zero == True:
            plt.axhline(0,color = 'black')
            
        if grid == True:
            plt.grid(True)
        
       
    fig.suptitle(root,y=1,fontweight="bold")
    plt.tight_layout()
    path.savefig(fig, bbox_inches = 'tight')

    plt.savefig('matplot_test.png')
    


start_year = 2005
end_year = 2020

fn = 'mar19_chartpack.pdf' #'dash_pdf.pdf'
pdf = PdfPages(fn)


def plotly_multi_plot(df,countries,root,pct,startYear = 1900,zero = False, filename = 'foo2',path='err', grid=False, just_month = False):
    startYear = int(startYear)
    #fig = plt.figure(figsize=(18,18))
    fig = plt.figure(figsize=(26,20))
    #gs =gridspec.GridSpec(8, 4)
    #gs =gridspec.GridSpec(12, 4)
    gs =gridspec.GridSpec(7, 4)

    #for i in range(0,len(countries)-1):
    for i in range(0,len(countries)):
        
        if len(df[countries[i]][df[countries[i]]<0]) > 0:
            zero = True
            #plt.autoscale(enable=True)
            #print(countries[i],plt.gca().get_ylim()[0])
            #print(df[countries[i]].min()*2)
            #plt.ylim((df[countries[i]].min()*2,plt.gca().get_ylim()[1]))
        else:
            zero = False
            #plt.ylim((-df[countries[i]].min()*.1,plt.gca().get_ylim()[1])) ####
        
        plt.subplot(gs[i])
        
        
        to_plt = df[countries[i]][df.index.year>=startYear]
        plt.plot(to_plt, color='blue',linewidth=0.75)
        plt.title('\n{}'.format(countries[i]))
        #plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        
        if just_month == True:
            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d-%b'))
            
        else:
            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m-%y'))
        
        if pct == True:
            plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter())

        #plt.ylim((-10,10))
        if zero == True:
            plt.axhline(0,color = 'black')
            
        if grid == True:
            plt.grid(True)
        
       
    fig.suptitle(root,y=1,fontweight="bold")
    plt.tight_layout()
    path.savefig(fig, bbox_inches = 'tight')
#     plt.savefig('matplot_test.png')
    test_fig= plt.gcf()
    return test_fig

test_graph = mpl_to_plotly(plotly_multi_plot(macro_ind.fillna(method='ffill')[str(2005):], countries=macro_ind.columns,root='Macro Indicators',pct=True, filename='Macro Indicators', path=pdf))

def plotly_multi_series_plot(df1,df2,df3,col_names,title = 'title',labels = ['series1','series2','series3'],startYear = 1900,zero = False, filename = 'foo2',path='err', grid=False, just_month = False):
    startYear = int(startYear)
    fig = plt.figure(figsize=(26,20))
    #gs =gridspec.GridSpec(8, 4)
    gs =gridspec.GridSpec(7, 4, figure = fig)
    
    #for i in range(0,len(countries)-1):
    for i in range(0,len(col_names)):
        plt.subplot(gs[i])
        
        if grid == True:
            plt.grid(True)
        
        if isinstance(df3, pd.DataFrame):
            chart_df = pd.DataFrame({
                'series1':df1['{}'.format(col_names[i])],
                'series2': df2['{}'.format(col_names[i])],
                'series3': df3['{}'.format(col_names[i])]
                
            }
            )#.fillna(method='bfill')

            plt.plot(chart_df[chart_df.index.year >= startYear],linewidth=0.75)
            
            if just_month == True:
                plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d-%b'))

            #cy = cycler('color', ['blue', 'red', 'blue'])
            #ax.set_prop_cycle(cy)
            
            plt.title('\n{}'.format(col_names[i]))
            if i==2:
                plt.legend(labels, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
                
            plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter())


            #plt.ylim((-10,10))
            if zero == True:
                plt.axhline(0,color = 'black')
                
                
        else:
            chart_df = pd.DataFrame({
                'series1':df1['{}'.format(col_names[i])],
                'series2': df2['{}'.format(col_names[i])]
            }
            )#.fillna(method='bfill')
            
            plt.plot(chart_df[chart_df.index.year >= startYear],linewidth=0.75)
            if just_month == True:
                plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d-%b'))
            
            #cycler('color', ['blue', 'red'])
            #ax.set_prop_cycle(cy)

            plt.title('\n{}'.format(col_names[i]))
            if i ==2:
                plt.legend(labels[:2], bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
            plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter())


            #plt.ylim((-10,10))
            if zero == True:
                plt.axhline(0,color = 'black')
    
    fig.suptitle(title,y=1,fontweight="bold")
    plt.tight_layout()
    path.savefig(fig, bbox_inches = 'tight')
    plotly_multi_series_fig = plt.gcf()
    return plotly_multi_series_fig

disc_18mo_change_in_3m = mpl_to_plotly(plotly_multi_plot(disc_tight.fillna(method='ffill')[str(start_year):str(end_year)]*100,cCodesRates,root = 'Discounted 18-month Change in 3m Rate',pct=True,startYear = start_year,zero = True,filename = 'discounted_change',path=pdf))
realized_18m_graph = mpl_to_plotly(plotly_multi_series_plot((shortRate - shortRate.shift(378)).fillna(method='ffill')[str(start_year):str(end_year)]*100, disc_tight_spot[str(start_year):str(end_year)]*100,0,cCodesRates, title = 'Realized 18-month Changes in 3m Rate vs. Currently Discounted 18-month Change in 3m Rate',labels = ['Realized','Disc. Change',0],startYear = start_year,zero = True, filename = 'historical_vs_realized_tightening',path=pdf))
yc_across_countries = mpl_to_plotly(plotly_multi_plot(yc[str(start_year):str(end_year)].fillna(method='ffill')*10000,cCodesRates,root = '5y-3m Yield Curve (bps)',pct=False,startYear = start_year,zero = True,filename = 'yc',path=pdf))
gdp_growth_measures = mpl_to_plotly(plotly_multi_series_plot(ewma(gs_house_extend[str(start_year):str(end_year)],2)*100,rgdp_growth[str(start_year):str(end_year)]*100,rpot_growth[str(start_year):str(end_year)]*100,cCodesRates+['EM','DM','Global'],title = 'Growth Measures',labels = ['House & CAI Avg','6m Annualized RGDP','Potential Real Growth'],startYear = start_year,zero = True, filename = 'growth',path=pdf))
inflation_and_core_inflation = mpl_to_plotly(plotly_multi_series_plot(inflation_core[str(start_year):str(end_year)],inflation[str(start_year):str(end_year)],0,cCodesRates+['EM','DM','Global'], title = 'Inflation',labels = ['Core','Headline',0],startYear = start_year,zero = True,filename = 'inflation',path=pdf))
current_account =mpl_to_plotly(plotly_multi_plot(get_ca_usd_ann(cCodesRates,['CZK','PHP'])[str(start_year):str(end_year)], countries = cCodesRates, root = 'Current Account (Ann. USD Billions)', pct = False,startYear = start_year,zero = True,filename = 'curr_acct',path=pdf))
citi_economic_surprise = mpl_to_plotly(plotly_multi_plot(ewma(citi_surprise[str(start_year):str(end_year)],10), countries = cCodesRates[:8] + ['EM','DM','World'], root = 'Citi Economic Surprise Index', pct = False,startYear = start_year,zero = True,filename = 'citi_surp',path=pdf))
fx_cum_90d_ret = mpl_to_plotly(plotly_multi_series_plot(spot_daily_rets[-89:].cumsum()*100,fx_daily_rets[-89:].cumsum()*100,0,cCodesRates,title = 'FX Cumulative 90-day Returns',labels = ['Spot Change','Total Return'],startYear = start_year,zero = True, filename = 'growth',grid = True, just_month = True, path = pdf))

em = ['IND','CHN','TLD','KOR','RUS','CZK','PLD','HUN','TUR','BRZ','MEX','CHI','COL','SAF','IDR','PHP']

eq_index_cum_90d_ret = mpl_to_plotly(plotly_multi_plot(equity_daily_rets[-89:].cumsum()*100, cCodesRates, root = 'Equity Index Cumulative 90-day Returns', pct = True, grid=True, just_month=True, filename = 'equity_rets', path = pdf))
short_rates = mpl_to_plotly(plotly_multi_series_plot(shortRate.loc[shortRate.drop('PHP',axis=1).dropna().index,:][2016:]*100,two[str(2016):],five[str(2016):],cCodesRates,title = 'Rates',labels = ['3M','2y','5y'],startYear = start_year, filename = 'growth',path=pdf, grid = True, just_month = True))

def single_plotly_multi_plot(df,countries,root,pct,startYear = 1900,zero = False, filename = 'foo2',path='err', grid=False, just_month = False):
    startYear = int(startYear)
    #fig = plt.figure(figsize=(18,18))
    fig = plt.figure(figsize=(26,20))
    #gs =gridspec.GridSpec(8, 4)
    #gs =gridspec.GridSpec(12, 4)
    gs =gridspec.GridSpec(7,4)

    #for i in range(0,len(countries)-1):
    for i in range(0,len(countries)):
        
        if len(df[countries[i]][df[countries[i]]<0]) > 0:
            zero = True
            #plt.autoscale(enable=True)
            #print(countries[i],plt.gca().get_ylim()[0])
            #print(df[countries[i]].min()*2)
            #plt.ylim((df[countries[i]].min()*2,plt.gca().get_ylim()[1]))
        else:
            zero = False
            #plt.ylim((-df[countries[i]].min()*.1,plt.gca().get_ylim()[1])) ####
        
        plt.subplot(gs[i])
        
        
        to_plt = df[countries[i]][df.index.year>=startYear]
        plt.plot(to_plt, color='blue',linewidth=0.75)
        plt.title('\n{}'.format(countries[i]))
        #plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        
        if just_month == True:
            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d-%b'))
            
        else:
            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m-%y'))
        
        if pct == True:
            plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter())

        #plt.ylim((-10,10))
        if zero == True:
            plt.axhline(0,color = 'black')
            
        if grid == True:
            plt.grid(True)
        
       
    fig.suptitle(root,y=1,fontweight="bold")
    plt.tight_layout()
    path.savefig(fig, bbox_inches = 'tight')
#     plt.savefig('matplot_test.png')
    test_fig= plt.gcf()
    return test_fig

def single_plotly_multi_series_plot(df1,df2,df3,col_names,title = 'title',labels = ['series1','series2','series3'],startYear = 1900,zero = False, filename = 'foo2',path='err', grid=False, just_month = False):
    startYear = int(startYear)
    fig = plt.figure(figsize=(26,20))
    #gs =gridspec.GridSpec(8, 4)
    gs =gridspec.GridSpec(7, 4, figure = fig)
    
    #for i in range(0,len(countries)-1):
    for i in range(0,len(col_names)):
        plt.subplot(gs[i])
        
        if grid == True:
            plt.grid(True)
        
        if isinstance(df3, pd.DataFrame):
            chart_df = pd.DataFrame({
                'series1':df1['{}'.format(col_names[i])],
                'series2': df2['{}'.format(col_names[i])],
                'series3': df3['{}'.format(col_names[i])]
                
            }
            )#.fillna(method='bfill')

            plt.plot(chart_df[chart_df.index.year >= startYear],linewidth=0.75)
            
            if just_month == True:
                plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d-%b'))

            #cy = cycler('color', ['blue', 'red', 'blue'])
            #ax.set_prop_cycle(cy)
            
            plt.title('\n{}'.format(col_names[i]))
            if i==2:
                plt.legend(labels, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
                
            plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter())


            #plt.ylim((-10,10))
            if zero == True:
                plt.axhline(0,color = 'black')
                
                
        else:
            chart_df = pd.DataFrame({
                'series1':df1['{}'.format(col_names[i])],
                'series2': df2['{}'.format(col_names[i])]
            }
            )#.fillna(method='bfill')
            
            plt.plot(chart_df[chart_df.index.year >= startYear],linewidth=0.75)
            if just_month == True:
                plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d-%b'))
            
            #cycler('color', ['blue', 'red'])
            #ax.set_prop_cycle(cy)

            plt.title('\n{}'.format(col_names[i]))
            if i ==2:
                plt.legend(labels[:2], bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
            plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter())


            #plt.ylim((-10,10))
            if zero == True:
                plt.axhline(0,color = 'black')
    
    fig.suptitle(title,y=1,fontweight="bold")
    plt.tight_layout()
    path.savefig(fig, bbox_inches = 'tight')
    plotly_multi_series_fig = plt.gcf()
    return plotly_multi_series_fig


class CountryVariables:
    country_list = ['USA', 'EUR', 'JPN', 'GBR', 'CAN', 'AUS', 'SWE', 'NZL', 'CHN', 'IND', 'KOR', 'TLD', 'TUR', 'RUS', 'PLD', 'CZK', 'HUN', 'BRZ', 'MEX', 'CHI', 'COL', 'SAF', 'IDR', 'PHP']

    eighteen_mo_change_in_3m = {country: mpl_to_plotly(single_plotly_multi_plot(disc_tight.fillna(method='ffill')[str(start_year):str(end_year)]*100,[country],root = 'Discounted 18-month Change in 3m Rate',pct=True,startYear = start_year,zero = True,filename = 'discounted_change',path=pdf)) for country in country_list}
    realized_18m = {country: mpl_to_plotly(single_plotly_multi_series_plot((shortRate - shortRate.shift(378)).fillna(method='ffill')[str(start_year):str(end_year)]*100, disc_tight_spot[str(start_year):str(end_year)]*100,0,[country], title = 'Realized 18-month Changes in 3m Rate vs. Currently Discounted 18-month Change in 3m Rate',labels = ['Realized','Disc. Change',0],startYear = start_year,zero = True, filename = 'historical_vs_realized_tightening',path=pdf)) for country in country_list}
    yc_across_countries = {country: mpl_to_plotly(single_plotly_multi_plot(yc[str(start_year):str(end_year)].fillna(method='ffill')*10000,[country],root = '5y-3m Yield Curve (bps)',pct=False,startYear = start_year,zero = True,filename = 'yc',path=pdf)) for country in country_list}
    gdp_growth_measures = {country: mpl_to_plotly(single_plotly_multi_series_plot(ewma(gs_house_extend[str(start_year):str(end_year)],2)*100,rgdp_growth[str(start_year):str(end_year)]*100,rpot_growth[str(start_year):str(end_year)]*100,[country],title = 'Growth Measures',labels = ['House & CAI Avg','6m Annualized RGDP','Potential Real Growth'],startYear = start_year,zero = True, filename = 'growth',path=pdf)) for country in country_list}
    inflation_and_core_inflation = {country: mpl_to_plotly(single_plotly_multi_series_plot(inflation_core[str(start_year):str(end_year)],inflation[str(start_year):str(end_year)],0,[country], title = 'Inflation',labels = ['Core','Headline',0],startYear = start_year,zero = True,filename = 'inflation',path=pdf)) for country in country_list}
    current_account = {country: mpl_to_plotly(single_plotly_multi_plot(get_ca_usd_ann([country], [])[str(start_year):str(end_year)], countries = [country], root = 'Current Account (Ann. USD Billions)', pct = False,startYear = start_year,zero = True,filename = 'curr_acct',path=pdf)) for country in country_list}
    current_account_czk_php = {country: mpl_to_plotly(single_plotly_multi_plot(get_ca_usd_ann([country], [country])[str(start_year):str(end_year)], countries = [country], root = 'Current Account (Ann. USD Billions)', pct = False,startYear = start_year,zero = True,filename = 'curr_acct',path=pdf)) for country in ['CZK', 'PHP']}
    # citi_economic_surprise = {country: mpl_to_plotly(single_plotly_multi_plot(ewma(citi_surprise[str(start_year):str(end_year)],10), countries = [country], root = 'Citi Economic Surprise Index', pct = False,startYear = start_year,zero = True,filename = 'citi_surp',path=pdf)) for country in country_list}
    fx_cum_90d_ret = {country: mpl_to_plotly(single_plotly_multi_series_plot(spot_daily_rets[-89:].cumsum()*100,fx_daily_rets[-89:].cumsum()*100,0,[country],title = 'FX Cumulative 90-day Returns',labels = ['Spot Change','Total Return'],startYear = start_year,zero = True, filename = 'growth',grid = True, just_month = True, path = pdf)) for country in country_list}
    eq_index_cum_90d_ret = {country: mpl_to_plotly(single_plotly_multi_plot(equity_daily_rets[-89:].cumsum()*100, [country], root = 'Equity Index Cumulative 90-day Returns', pct = True, grid=True, just_month=True, filename = 'equity_rets', path = pdf)) for country in country_list}
    short_rates = {country: mpl_to_plotly(single_plotly_multi_series_plot(shortRate.loc[shortRate.drop('PHP',axis=1).dropna().index,:][2016:]*100,two[str(2016):],five[str(2016):],[country],title = 'Rates',labels = ['3M','2y','5y'],startYear = start_year, filename = 'growth',path=pdf, grid = True, just_month = True)) for country in country_list}

    # ind_citi_economic_surprise = mpl_to_plotly(single_plotly_multi_plot(ewma(citi_surprise[str(start_year):str(end_year)],10), countries = ['IND'], root = 'Citi Economic Surprise Index', pct = False,startYear = start_year,zero = True,filename = 'citi_surp',path=pdf))
    czk_current_account =mpl_to_plotly(single_plotly_multi_plot(get_ca_usd_ann(['CZK'], ['CZK'])[str(start_year):str(end_year)], countries = ['CZK'], root = 'Current Account (Ann. USD Billions)', pct = False,startYear = start_year,zero = True,filename = 'curr_acct',path=pdf))
    php_current_account =mpl_to_plotly(single_plotly_multi_plot(get_ca_usd_ann(['PHP'], ['PHP'])[str(start_year):str(end_year)], countries = ['PHP'], root = 'Current Account (Ann. USD Billions)', pct = False,startYear = start_year,zero = True,filename = 'curr_acct',path=pdf))
 
