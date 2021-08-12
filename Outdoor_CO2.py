# -*- coding: utf-8 -*-
"""
Created on Tue Apr 20 16:28:54 2021

@author: Devineni
"""

import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime as dt, timedelta
from sqlalchemy import create_engine

from easygui import *

def prRed(skk): print("\033[31;1;m {}\033[00m" .format(skk)) 
def prYellow(skk): print("\033[33;1;m {}\033[00m" .format(skk)) 
# engine = create_engine("mysql+pymysql://wojtek:Password#102@wojtek.mysql.database.azure.com/",\
#                       pool_pre_ping=True) # Cloud server address
engine = create_engine("mysql+pymysql://root:Password123@localhost/",pool_pre_ping=True)
#%%
#%% control plot properties
import datetime
import matplotlib.dates as mdates

import matplotlib.units as munits
from pylab import rcParams
rcParams['figure.figsize'] = 10,4.5
plt.rcParams["font.family"] = "calibri"
plt.rcParams["font.weight"] = "normal"
plt.rcParams["font.size"] = 10

#%%

def strtime(string):
    """insert a string in regular python format to convert into a timestamp for easy evaluation"""
    return dt.strptime(string, '%Y-%m-%d %H:%M:%S')


#%%
def outdoor(*times, plot=False):
    """
    Syntax: 
        t0 (string, optional),tn (string, optional), plot = True or False (default False)
    Parameters
    ----------
    *times : time string in standard python datetime format
        if no time strings are provided user can select times from available experiments.
    plot : boolean, optional
        input True if the graph for this time period is required. The default is False.

    Returns
    -------
    df : dataframe 
        The columns are (['datetime', 'temp_°C', 'RH_%rH', 'CO2_ppm']
    experiment : string
        Either the experiment name or the database is returned depending on selection.

    """
    if len(times)==2:
        df = pd.read_sql_query("SELECT * FROM testdb.database_selection;", con = engine)

        if (strtime(times[0]) >= df["start"][0] ) & (strtime(times[1]) <= df["end"][0] ):
            database = df["database"][0]
        elif (strtime(times[0]) >= df["start"][1] ) & (strtime(times[1]) <= df["end"][1] ):
            database = df["database"][1]
        elif (strtime(times[0]) >= df["start"][2] ) & (strtime(times[1]) <= df["end"][2] ):
            database = df["database"][2]
        elif (strtime(times[0]) >= df["start"][3] ) & (strtime(times[1]) <= df["end"][3] ):
            database = df["database"][3]
            
        df = pd.read_sql_query("SELECT * FROM {}.außen WHERE datetime BETWEEN '{}' AND '{}';".format(database, times[0], times[1]), con = engine)
        df = df.loc[:, ['datetime', 'temp_°C', 'RH_%rH', 'CO2_ppm']]
        experiment = database
    else:
        times = pd.read_sql_query("SELECT * FROM testdb.times;", con = engine)
        
        msg ="Then, please select a timeframe to extract outdoor data"
        title = "Time selection for outdoor data extraction"
        choices = times.loc[:,"short_name"]
        experiment = choicebox(msg, title, choices)
        t0 = str(times.loc[times["short_name"] == experiment]["Start"].iat[0])
        tn = str(times.loc[times["short_name"] == experiment]["End"].iat[0])
        database = str(times.loc[times["short_name"] == experiment]["database"].iat[0])
    
    
        df = pd.read_sql_query("SELECT * FROM {}.außen WHERE datetime BETWEEN '{}' AND '{}';".format(database, t0, tn), con = engine)
        df = df.loc[:, ['datetime', 'temp_°C', 'RH_%rH', 'CO2_ppm']]
    
    if plot==True:
        title = "Outdoor air properties during {}".format(experiment)
        result = df.set_index("datetime")
    
        def make_patch_spines_invisible(ax):
            ax.set_frame_on(True)
            ax.patch.set_visible(False)
            for sp in ax.spines.values():
                sp.set_visible(False)
    
    
        fig, host = plt.subplots()
        fig.subplots_adjust(right=0.75)
    
        par1 = host.twinx()
        par2 = host.twinx()
    
        # Offset the right spine of par2.  The ticks and label have already been
        # placed on the right by twinx above.
        par2.spines["right"].set_position(("axes", 1.1))
        # Having been created by twinx, par2 has its frame off, so the line of its
        # detached spine is invisible.  First, activate the frame but make the patch
        # and spines invisible.
        make_patch_spines_invisible(par2)
        # Second, show the right spine.
        par2.spines["right"].set_visible(True)
    
        p1, = host.plot(result.index, result['temp_°C'], "b-", label="Temperature (°C)")
        p2, = par1.plot(result.index, result['CO2_ppm'], "r-", label="CO2 (ppm)")
        p3, = par2.plot(result.index, result['RH_%rH'], "g-", label="RH (%)")
    
        # host.set_xlim(0, 2)
        # host.set_ylim(0, 2)
        # par1.set_ylim(0, 4)
        # par2.set_ylim(1, 65)
    
        host.set_xlabel("Time")
        host.set_ylabel("Temperature (°C)")
        par1.set_ylabel("CO2 (ppm)")
        par2.set_ylabel("RH (%)")
    
        host.yaxis.label.set_color(p1.get_color())
        par1.yaxis.label.set_color(p2.get_color())
        par2.yaxis.label.set_color(p3.get_color())
    
        tkw = dict(size=4, width=1.5)
        host.tick_params(axis='y', colors=p1.get_color(), **tkw)
        par1.tick_params(axis='y', colors=p2.get_color(), **tkw)
        par2.tick_params(axis='y', colors=p3.get_color(), **tkw)
        host.tick_params(axis='x', **tkw)
    
        lines = [p1, p2, p3]
    
        plt.title(title)
    
        host.legend(lines, [l.get_label() for l in lines], loc = 3)
    
        plt.savefig(title + '.png', bbox_inches='tight', dpi=400)
    
        plt.show()
        
    return df, experiment
#%%


#%%
# outdoor("2020-02-26 03:26:00", "2020-02-26 08:30:00", plot = True)
# a, b=outdoor("2020-01-30 12:12:40", "2020-01-30 15:59:55", plot = False)







