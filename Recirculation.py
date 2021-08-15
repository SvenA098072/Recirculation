# -*- coding: utf-8 -*-
"""
Created on Thu Apr  8 14:26:27 2021

@author: Devineni and Sven finally merged
"""
# Necessary modules
import pandas as pd
pd.set_option('mode.chained_assignment',None)
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
pd.options.plotting.backend = "matplotlib" # NOTE: This is useful in case the plotbackend has been changed by any previously (even befor machine shut-downs).

# from statistics import mean
from tabulate import tabulate
from sqlalchemy import create_engine
from uncertainties import ufloat
from uncertainties import unumpy
from uncertainties import umath

from post_processing import CBO_ESHL


# functions to print in colour
def prRed(skk): print("\033[31;1;m {}\033[00m" .format(skk)) 
def prYellow(skk): print("\033[33;1;m {}\033[00m" .format(skk)) 
# The following is a general syntax to dedine a MySQL connection
engine = create_engine("mysql+pymysql://root:Password123@localhost/",pool_pre_ping=True)
### =============================================================================
### engine = create_engine("mysql+pymysql://admin:the_secure_password_4_ever@localhost/",\
###                           pool_pre_ping=True) # Krishna's local address
###=============================================================================
### engine = create_engine("mysql+pymysql://wojtek:Password#102@wojtek.mysql.database.azure.com/",\
###                       pool_pre_ping=True) # Cloud server address

class ResidenceTimeMethodError(ValueError):
    def __str__(self):
        return 'You need to select a valid method: iso, trapez or simpson (default)'
    
#%% Function shape to be fitted for infinity concentration of tau_exh
def expfitshape(x, a, b):
    return a*x*np.exp(-x/b)

#%% Function to find outliers
def find_outliers(col):
    from scipy import stats
    z = np.abs(stats.zscore(col))
    idx_outliers = np.where(z>3,True,False)
    return pd.Series(idx_outliers,index=col.index)


def residence_time_sup_exh(experiment='W_I_e0_ESHL', aperture_sensor = "2l", periodtime=120, 
                           experimentname=False, plot=False, 
                           export_sublist=False, method='simpson',
                           filter_maxTrel=0.25, logging=False):
    """
        method:
            'iso'       (Default) The method described in ISO 16000-8 will be applied
                        however this method has a weak uncertainty analysis.
            'trapez'    corrected ISO 16000-8 method applying the trapezoidal method
                        for the interval integration and considers this in the 
                        uncertainty evaluation.
            'simpson'   Applies the Simpson-Rule for the integration and consequently 
                        considers this in the uncertainty evaluation.
        
        filter_maxTrel:
            Percentage value for the allowed deviation of the predefined 
            periodtime T of the devices. Only half-cycles which meet the 
            criterion ]T/2*(1-filter_maxTrel),T/2*(1+filter_maxTrel)[
            are going to be evaluated.
    """    
    
    #%% Function import  
    """Syntax to import a function from any folder. Useful if the function.py file 
       is in another folder other than the working folder"""
    # import sys  
    # import sys  
    # sys.path.append("C:/Users/Devineni/OneDrive - bwedu/4_Recirculation/python_files/")  
    from Outdoor_CO2 import outdoor # This function calculates the outdoor CO2 data
    global a, b, df_tau_sup, df_tau_exh
    #%% Control plot properties"
    """This syntax controls the plot properties(default plot font, shape, etc), 
        more attributes can be added and removed depending on the requirement """
    
    from pylab import rcParams
    rcParams['figure.figsize'] = 7,4.5
    plt.rcParams["font.family"] = "calibri"
    plt.rcParams["font.weight"] = "normal"
    plt.rcParams["font.size"] = 10
    	
    plt.close("all")
    
    
    
    if periodtime is None:
        T = 120
        prYellow('ATTENTION: periodtime has not been defined. I setted T=120s instead')
    else:
        T = periodtime
    
    # T in s; period time of the ventilation systems push-pull devices.
    # time = pd.read_excel("C:/Users/Devineni/OneDrive - bwedu/4_Recirculation/Times_thesis.xlsx", sheet_name="Timeframes")
    # The dataframe time comes from the excel sheet in the path above, to make -
    # - changes go to this excel sheet, edit and upload it to mysql.
    
    lb = T/2*(1-filter_maxTrel) # lower bound of considered cycles
    ub = T/2*(1+filter_maxTrel) # upper bound of considered cycles
    
    time = pd.read_sql_query("SELECT * FROM testdb.timeframes;", con = engine)      
    #standard syntax to fetch a table from Mysql; In this case a table with the 
    # short-names of the measurements, all the start and end times, the DB-name 
    # of the measurement and the required table-names of the DB/schema is loaded into a dataframe. 
    
    #%% Load relevant data
    t = time.index[time['short_name'].isin([experiment])==True].tolist()[0] # to select the experiment (see Timeframes.xlsx)
    
    start = str(time["Start"][time.index[time['short_name'].isin([experiment])==True].tolist()[0]] - dt.timedelta(minutes=20))
    end = str(time["End"][time.index[time['short_name'].isin([experiment])==True].tolist()[0]])

    t0 = time["Start"][t]
    # actual start of the experiment, out of the dataframe "time"
        
     
    table = time["tables"][t].split(",")                                         #Name of the ventilation device
    try: 
        if aperture_sensor in aperture_sensor:
            pass
        else:
            raise ValueError
    except ValueError:
        prYellow('ValueError: The sensor you selected is not an aperture sensor of the experiment. Select one of these: {}'.format(table))
        return 'ValueError: The sensor you selected is not an aperture sensor of the experiment. Select one of these: {}'.format(table)
    
    dum = [["Experiment", experiment ], ["Sensor", aperture_sensor]]                # Creates a list of 2 rows filled with string tuples specifying the experiment and the sensor.
    if experimentname:
        print(tabulate(dum))                                                            # Prints the inut details in a table
    else:
        pass
    
    database = time["database"][time.index[time['short_name'].isin([experiment])==True].tolist()[0]]    # Selects the name of the database as a string 
    
    #%%% Load background data
       
    background, dummy = outdoor(str(t0), str(end), plot = False)                    # Syntax to call the background concentration function, "dummy" is only necessary since the function "outdoor" returns a tuple of a dataframe and a string.
    background = background["CO2_ppm"].mean()                                       # Future: implement cyclewise background concentration; Till now it takes the mean outdoor concentration of the whole experiment.

    #%%% Load data of the experiment and the selected sensor
    
    df = pd.read_sql_query("SELECT * FROM {}.{} WHERE datetime BETWEEN '{}' AND\
                           '{}'".format(database, aperture_sensor, start, end), con = engine)
    df = df.loc[:,["datetime", "CO2_ppm"]]
    df["original"] = df["CO2_ppm"]                                                  # Copies the original absolute CO2-concentrations data form CO2_ppm in a "backup"-column originals 
    df.columns = ["datetime", "original", "CO2_ppm"]                                # changes the order of the columns to the defined one
    
    # df["original"] = df["CO2_ppm"]                                                  # Copies the original absolute CO2-concentrations data form CO2_ppm in a "backup"-column originals; this one can be deleted 
    
    df["CO2_ppm"] = df["CO2_ppm"] - background                                      # substracts the background concentrations -> CO2_ppm contains CO2-concentration of some instance of time above background concentration.
    
    if df["CO2_ppm"].min() < 0:                                                     # Sometimes the accumulated amount of CO2 concentraion becomes negative. This is not possible and would lead to a mistake for the integral calculation. An artificial offset lifts the whole decay curve at >=0.
        offset = df["CO2_ppm"].min()
        df["CO2_ppm"] = df["CO2_ppm"] - offset
        
    df = df.loc[~df.duplicated(subset=["datetime"])]                                # Checks for duplicated in datetime and removed them; @Krishna: How can such a duplicate occur?
    diff = (df["datetime"][1]-df["datetime"][0]).seconds                            # integer diff in s; Calculates the length of the time interval between two timestamps 
    df = df.set_index("datetime")                                                   # Resets the index of the dataframe df from the standard integer {0, 1, 2, ...} to be exchanged by the datetime column containing the timestamps.
    
    t01 = t0
    while not(t01 in df.index.to_list()):                                            # The t0 from the excel sheet may not be precice that the sensor starts 
        t01 = t01 + dt.timedelta(seconds=1)                                 # - starts at the same time so i used this while loop to calculate the 
                                                                          # - the closest t0 after the original t0
    
    df["roll"] = df["CO2_ppm"].rolling(int(T/diff)).mean()                          # moving average for 2 minutes, used to calculate Cend; T = 120s is the period time of the push-pull ventilation devices which compose the ventilation system. 
    df["roll"] = df["roll"].fillna(method='bfill')
    
    c0 = df["roll"].loc[t01]                                                      # C0; @DRK: Check if c0 = df["roll"].loc[t0] is better here. ## ORIGINAL: c0 = df["CO2_ppm"].loc[t0] 
    Cend37 = round((c0)*0.37, 2)  
    df2 = df                                                  # @DRK: From this line 101 schould be changed.   
    
    cend = df.loc[df2["roll"].le(Cend37)]                                            # Cend: Sliced df of the part of the decay curve below the 37 percent limit
    tn = df.index[-1]
    
    if len(cend) == 0:                                                              # Syntax to find the tn of the experiment
        print("The device has not reached 37% of its initial concentration")
    else:
        pass
    
    
    #%%% Increase resolution
    
    df = df.resample("5S").mean()
       
    df['original'] = df['original'].interpolate(method='polynomial', limit_direction='forward',order=2)
    df['CO2_ppm'] = df['CO2_ppm'].interpolate(method='polynomial', limit_direction='forward',order=2)
    df['roll'] = df['roll'].interpolate(method='polynomial', limit_direction='forward',order=2)
    
    #%%% Find max min points
    from scipy.signal import argrelextrema                                          # Calculates the relative extrema of data.
    n = round(T / (2*diff))                                                         # How many points on each side to use for the comparison to consider comparator(n, n+x) to be True.; @DRK: This value should depend on diff and T (period time of the push-pull devices). n = T / (2*diff)
    
    df['max'] = df.iloc[argrelextrema(df['CO2_ppm'].values, np.greater_equal,\
                                      order=n)[0]]['CO2_ppm']                       # Gives all the peaks; "np.greater_equal" is a callable function which argrelextrema shall use to compare to arrays before and after the point currently evaluated by argrelextrema.
    df['min'] = df.iloc[argrelextrema(df['CO2_ppm'].values, np.less_equal,\
                                      order=n)[0]]['CO2_ppm']                       # Gives all the valleys; "np.less_equal" is a callable function which argrelextrema shall use to compare to arrays before and after the point currently evaluated by argrelextrema. 
        
    #%%% Plot Original
    
    if plot:
        fig,ax = plt.subplots()
        df.plot(title = "original " + experiment, color = [ 'silver', 'green', 'orange'], ax = ax)
        df['max'].plot(marker='o', ax = ax)                                             # This needs to be verified with the graph if python recognizes all peaks
        df['min'].plot(marker="v", ax = ax)                                             # - and valleys. If not adjust the n value.
    else:
        pass
    
    #%%% Load data for the occupied space V3
    
    experimentglo = CBO_ESHL(experiment = dum[0][1], aperture_sensor = aperture_sensor)
    
    alpha_mean, df_alpha, df_indoor = experimentglo.mean_curve()
    alpha_mean_u = ufloat(alpha_mean[0], alpha_mean[1])
    
    dfin_dCmean = df_indoor.loc[:,['mean_delta', 'std mean_delta']]
    tmeancurve = dfin_dCmean.index.tolist()[0]
    while not(tmeancurve in df.index.to_list()):                                            # The t0 from the excel sheet may not be precice that the sensor starts 
        tmeancurve = tmeancurve + dt.timedelta(seconds=1)
    datetime_index = pd.date_range(tmeancurve, dfin_dCmean.index.tolist()[-1], freq='5s')
    dfin_dCmean = dfin_dCmean.reindex(datetime_index, method='bfill')
    
    if t0 < dfin_dCmean.index.tolist()[0]:
        mean_delta_0_room  = dfin_dCmean.loc[dfin_dCmean.index.tolist()[0]]
        deltat_mean = dfin_dCmean.index.tolist()[0] - t0
        prYellow('ATTENTION: mean_delta_room starts {} after t0 = {}!'.format(deltat_mean, t0))
    else:
        mean_delta_0_room = dfin_dCmean.loc[t0]
    mean_delta_0_room_u = ufloat(mean_delta_0_room[0],mean_delta_0_room[1])
   
    #%%%%% Add mean and exhaust concentrations indoor (V3) to the dfin_dCmean
    '''
        mean concentrations: 
            Based on the calculated spatial and statistical mean air
            age in the occupied space and the spacial average initial 
            concentration in the occupied space at t0.
    '''
    
    # count = 0
    dfin_dCmean['room_av'] = pd.Series(dtype='float64')
    dfin_dCmean['std room_av'] = pd.Series(dtype='float64')
    dfin_dCmean['room_exh'] = pd.Series(dtype='float64')
    dfin_dCmean['std room_exh'] = pd.Series(dtype='float64')
    dfin_dCmean.reset_index(inplace=True)
    if 'index' in dfin_dCmean.columns:
        dfin_dCmean = dfin_dCmean.rename(columns={"index": "datetime"})
    else:
        pass
    
    for count in range(len(dfin_dCmean)):     
        deltat = dfin_dCmean['datetime'][count]-t0
        deltat = deltat.total_seconds()/3600
        '''
        mean concentrations: 
            Based on the calculated spatial and statistical mean air
            age in the occupied space and the spacial average initial 
            concentration in the occupied space at t0.
        '''    
        value = mean_delta_0_room_u*umath.exp(-1/(alpha_mean_u)*deltat)
        dfin_dCmean['room_av'][count] = value.n
        dfin_dCmean['std room_av'][count] = value.s
        
        '''
        exhaust concentrations: 
            Based on the calculated spatial and statistical mean air
            age in the occupied space and the spacial average initial 
            concentration in the occupied space at t0.
        '''
        value = mean_delta_0_room_u*umath.exp(-1/(2*alpha_mean_u)*deltat)
        dfin_dCmean['room_exh'][count] = value.n
        dfin_dCmean['std room_exh'][count] = value.s
        
        # count = count + 1
    
    dfin_dCmean = dfin_dCmean.set_index('datetime')           
        
    
    #%%% Filter supply and exhaust phases 
    df.loc[df['min'] > -400, 'mask'] = False                                        # Marks all min as False; @DRK: Why is this "-400" necessary?                         
    df.loc[df['max'] > 0, 'mask'] = True                                            # Marks all max as True; @DRK: This is just a back-up right? For the case I use for debugging there is no change happening for df.
    df["mask"] = df["mask"].fillna(method='ffill').astype("bool")                   # Use forward to fill True and False 
    df = df.dropna(subset= ["mask"])                                                # In case there are NaNs left (at the beginning of the array) it drops/removes the whole time stamps/rows.
    df["sup"] = df["mask"]                                                          # Create seperate columns for sup and exhaust; @DRK: Why is this necessary? At the end of these six lines of code df has 3 column {mask, sup, exh} containing all there the same data.
    df["exh"] = df["mask"]
        
    
    df.loc[df['min'] > 0, 'sup'] = True                                             # The valleys have to be belong to supply as well 
    df.loc[df['max'] > 0, 'exh'] = False                                            # The peaks have to belong to max, before it was all filled be backfill
    
    
    df_sup = df.loc[df["sup"].to_list()]                                            # Extract all the supply phases form df. Meaning only the timestamps maeked with "True" in df["sup"] are selected. 
    
    a = df_sup.resample("5S").mean()                                                # Resampled beacuase, the time stamps are missing after slicing out the supply phases form df. The option "5S" adds the now missing time stamps again but without data. This is only necessary to plot the arrays flawlessly later in the same graphs again. 
    
    
    df_sup2 = a.loc[:,["CO2_ppm"]]
    
    df_exh = df.loc[~df["exh"].values]
    b = df_exh.resample("5S").mean()
    df_exh2 = b.loc[:,["CO2_ppm"]]
    
    sup_exh_df = pd.concat([dfin_dCmean, df_sup2, df_exh2], axis = 1).reset_index()
    sup_exh_df.columns = ["datetime", 
                              "meas room_av", "std meas room_av", 
                              "calc room_av", "std calc room_av",  
                              "calc room_exh", "std calc room_exh", 
                              "supply", "exhaust"]
    rows = sup_exh_df[~sup_exh_df['calc room_exh'].isnull()].index.tolist()
    sup_exh_df['d calc exh-av'] = np.sqrt(np.power(sup_exh_df["calc room_exh"].loc[rows],2)\
                                          - np.power(sup_exh_df["calc room_av"].loc[rows],2))
    sup_exh_df['std d calc exh-av'] = np.sqrt(np.power(sup_exh_df["std calc room_exh"].loc[rows],2)\
                                              + np.power(sup_exh_df["std calc room_av"].loc[rows],2))
    
    #%%%%%% Calculation of the weight factor of the current device period
            
    ddCmax_exhav = sup_exh_df.loc[sup_exh_df['d calc exh-av'].idxmax()]
    ddCmax_exhav = ddCmax_exhav.filter(['datetime','d calc exh-av','std d calc exh-av'])
    
    #%%% Plot Matplotlib                                                            # This can be verified from this graph        
# =============================================================================
#     if plot:
#         #%%%% supply
#         plt.figure()
#         a["CO2_ppm"].plot(title = "supply " + experiment) 
#         a["CO2_ppm"].plot(title = "supply") 
#           
#         #%%%% exhaust
#         b["CO2_ppm"].plot(title = "exhaust " + experiment)                                            # Similar procedure is repeated from exhaust
#         plt.figure()
#         b["CO2_ppm"].plot(title = "exhaust")                                            # Similar procedure is repeated from exhaust
#         
#         #%%%% Plot for extra prespective
#         fig,ax1 = plt.subplots()
#         
#         df_sup.plot(y="CO2_ppm", style="yv-", ax = ax1, label = "supply")
#         df_exh.plot(y="CO2_ppm", style="r^-", ax = ax1, label = "exhaust")
#     else:
#         pass
# =============================================================================
    
    #%%% Plot Plotly
    if plot:
        pd.options.plotting.backend = "plotly" # NOTE: This changes the plot backend which should be resetted after it is not needed anymore. Otherwise it will permanently cause problems in future, since it is a permanent change.
    
        import plotly.graph_objects as go
        fig = go.Figure()
        fig.add_trace(go.Scatter(#title = time["short_name"][t],
                                 name='meas room_av',
                                 x = sup_exh_df["datetime"], 
                                 y = sup_exh_df["meas room_av"],
                                 #error_y=dict(value=sup_exh_df["std meas room_av"].max())
                                 )
                      )
        fig.add_trace(go.Scatter(name='calc room_av',
                                 x = sup_exh_df["datetime"], 
                                 y = sup_exh_df["calc room_av"],
                                 #error_y = dict(value=sup_exh_df["std calc room_av"].max())
                                 )
                      )
        fig.add_trace(go.Scatter(name='calc room_exh',
                                 x = sup_exh_df["datetime"], 
                                 y = sup_exh_df["calc room_exh"],
                                 #error_y=dict(value=sup_exh_df["std calc room_exh"].max())
                                 )
                      )
        fig.add_trace(go.Scatter(name='d calc exh-av',
                                 x = sup_exh_df["datetime"], 
                                 y = sup_exh_df["d calc exh-av"],
                                 #error_y=dict(value=sup_exh_df["std d calc exh-av"].max())
                                 )
                      )
        fig.add_trace(go.Scatter(name='supply',x=sup_exh_df["datetime"], y = sup_exh_df["supply"]))
        fig.add_trace(go.Scatter(name='exhaust',x=sup_exh_df["datetime"], y = sup_exh_df["exhaust"]))
        
        fig.update_layout(
            title="{} {}".format(database, aperture_sensor),
            xaxis_title="Zeit t in hh:mm:ss",
            yaxis_title=r'Verweilzeit $\bar{t}_1$',
            legend_title="Legende",
            font=dict(
                family="Segoe UI",
                size=18,
                color="black"
            )
        )
        
        fig.show()
        
        import plotly.io as pio
        
        pio.renderers.default='browser'
        pd.options.plotting.backend = "matplotlib" # NOTE: This is a reset and useful in case the plotbackend has been changed by any previously (even befor machine shut-downs).
    else:
        pass
    
    #%% Marking dataframes supply
    """Marks every supply dataframe with a number for later anaysis """
    n = 1
    df_sup3 = df_sup2.copy().reset_index()                                          
    
    start_date = str(t0); end_date = str(tn) # CHANGE HERE 
    
    mask = (df_sup3['datetime'] > start_date) & (df_sup3['datetime'] <= end_date)
    
    df_sup3 = df_sup3.loc[mask]
    
    
    for i,j in df_sup3.iterrows():                                                  
        # *.interrows() will always return a tuple encapsulating an int for the 
        # index of the dataframe where it is applied to and a series containing 
        # the data of row selected. Therefore it is good to seperate both before in e.g. i,j .
        try:
            # print(not pd.isnull(j["CO2_ppm"]), (np.isnan(df_sup3["CO2_ppm"][i+1])))
            if (not pd.isnull(j["CO2_ppm"])) and (np.isnan(df_sup3["CO2_ppm"][i+1])):
                df_sup3.loc[i,"num"] = n
                n = n+1
            elif (not pd.isnull(j["CO2_ppm"])):
                df_sup3.loc[i,"num"] = n
        except KeyError:
            pass
            # print("ignore the key error")
        
    #%%%% Exporrt a file with all the supply curves sorted in a matrix for an excel diagram    
    df_sup_list = []
    dummy_df = pd.DataFrame(columns=['datetime', 'CO2_ppm', 'num'])
    for i in range(1, int(df_sup3['num'].max()+1)):

        try:
            if export_sublist and len(df_sup3.loc[df_sup3["num"]==i]) > 3:
                dummy_df = dummy_df.append(df_sup3.loc[df_sup3["num"]==(i)])
                dummy_df = dummy_df.rename(columns = {'CO2_ppm':'CO2_ppm_{}'.format(i)})
            
            
        except KeyError:
            pass
            # print("ignore the key error")

        df_sup_list.append(df_sup3.loc[df_sup3["num"]==i])
    
    del dummy_df["num"]
    if logging:
        dummy_df.to_csv(r'D:\Users\sauerswa\wichtige Ordner\sauerswa\Codes\Python\Recirculation\export\df_sup_{}_{}.csv'.format(database, aperture_sensor), index=True) 
    
    #%%% Supply tau 
    # This method can be replicated in excel for crossreference
    """Calculates tau based in ISO 16000-8"""
    
    
    if (database == "cbo_summer") or (database == "cbo_winter") or (database == "eshl_winter"):
        engine1 = create_engine("mysql+pymysql://root:Password123@localhost/{}".format("cbo_calibration"),pool_pre_ping=True)
#        engine = create_engine("mysql+pymysql://root:@34.107.104.23/{}".format("cbo_calibration"),pool_pre_ping=True)

    elif database == "eshl_summer":
        engine1 = create_engine("mysql+pymysql://root:Password123@localhost/{}".format("eshl_calibration"),pool_pre_ping=True)
#        engine = create_engine("mysql+pymysql://root:@34.107.104.23/{}".format("eshl_calibration"),pool_pre_ping=True)

    else:
        print("Please select a correct database")
    
    
    reg_result = pd.read_sql_table("reg_result", con = engine1).drop("index", axis = 1)
    '''Calibration data for the particular sensor alone is filtered '''
    global res

    res = reg_result[reg_result['sensor'].str.lower() == aperture_sensor].reset_index(drop = True) # Contains the sensor calibration data and especially the calibration curve.
    accuracy1 = 50 # it comes from the equation of uncertainity for testo 450 XL
    accuracy2 = 0.02 # ±(50 ppm CO2 ±2% of mv)(0 to 5000 ppm CO2 )
            
    accuracy3 = 50 # the same equation for second testo 450 XL
    accuracy4 = 0.02
            
    accuracy5 = 75 # # the same equation for second tes
    accuracy6 = 0.03 # Citavi Title: Testo AG
    
    df_tau_sup = []
    s_rel_start = 1-df_sup_list[0].reset_index()['CO2_ppm'].loc[0]/mean_delta_0_room['mean_delta']
    s_cyc = 0
    for idf in df_sup_list:
        if len(idf) > 3:
            a = idf.reset_index(drop = True)                                    # Overwride the dummy dataframe "a" by the currently chosen supply decay curve.
            a['CO2_ppm_reg'] = a.eval(res.loc[0, "equation"])                   # See: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.eval.html?highlight=pandas%20dataframe%20eval#pandas.DataFrame.eval    
            a = a.rename(columns = {'CO2_ppm':'CO2_ppm_original', 'CO2_ppm_reg': 'CO2_ppm'})
            a = a.drop_duplicates(subset=['datetime'])
            a = a.loc[:, ["datetime", "CO2_ppm_original", "CO2_ppm"]]
                        
            diff = (a["datetime"][1] - a["datetime"][0]).seconds
            
            a["runtime"] = np.arange(0,len(a) * diff, diff)
            
            lena = a["runtime"].iloc[-1]
            
            if a["runtime"].iloc[-1]>T/2*(1-filter_maxTrel) and \
                a["runtime"].iloc[-1]<T/2*(1+filter_maxTrel):
                    
                if logging:
                    prYellow("Since: {} < {} < {}, I consider the supply cycle {}".format(lb, lena, ub, s_cyc))
            
                ### Calculating the measurement uncertainty based on the uncertainties of the reference sensors and the deviation of the sensor during its calibration
                a["s_meas"] =  np.sqrt(np.square((a["CO2_ppm"] * accuracy2)) 
                                       + np.square(accuracy1) + np.square((a["CO2_ppm"] * accuracy4)) 
                                       + np.square(accuracy3) + np.square((a["CO2_ppm"] * accuracy6)) 
                                       + np.square(accuracy5)+ np.square(res.loc[0, "rse"]))
                ns_meas = a['s_meas'].mean()
                n = len(a['s_meas'])
                
                global sa_num, s_lambda, s_phi_e
                global area_sup, s_rest, s_total, a_rest, a_tot,sa_num,s_lambda, s_phi_e,s_rest, sa_rest, s_area
                 
                a = a.dropna()
                a = a[a["CO2_ppm"] >= a["CO2_ppm"].iloc[0]*0.36]
                a["log"] = np.log(a["CO2_ppm"])
                a = a.dropna()
                                        
                ### ISO 16000-8 option to calculate slope (defined to be calculated by Spread-Sheat/Excel)
                a["t-te"] = a["runtime"] - a["runtime"][len(a)-1]
                
                a["lnte/t"] = a["log"][len(a)-1] - a["log"]                         # @DRK: The slope (as defined in ISO 16000-8) was always negative since the two subtrahend where in the wrong order.
                
                a["slope"] = a["lnte/t"] / a["t-te"]
                
                try:
                    if method=='iso':
                        slope = a["slope"].mean()
                        
                        sumconz = a["CO2_ppm"].iloc[1:-1].sum()
                        area_sup = (diff * (a["CO2_ppm"][0]/2 + sumconz +a["CO2_ppm"][len(a)-1]/2))
                        print('ATTENTION: ISO 16000-8 method has a weak uncertainty evaluation consider using trapezoidal method is correcting this.')
                        
                    elif method=='trapez':
                        ### More acurate option to calculate the solpe of each (sub-)curve
                        x1 = a["runtime"].values
                        y1 = a["log"].values
                        
                        from scipy.stats import linregress
                        slope = -linregress(x1,y1)[0]
                        
                        from numpy import trapz
                        area_sup = trapz(a["CO2_ppm"].values, dx=diff) # proof that both methods have same answer:  area_sup_2 = area_sup_1
                        print('ATTENTION: Trapezoidal method is used in ISO 16000-8 and here also considered in the uncertainty evaluation. However, more precise results are given by applying the Simpson-Rule.')
                    
                    elif method=='simpson':
                        ### More acurate option to calculate the solpe of each (sub-)curve
                        x1 = a["runtime"].values
                        y1 = a["log"].values
                        
                        from scipy.stats import linregress
                        slope = -linregress(x1,y1)[0]
                        
                        from scipy.integrate import simpson
                        area_sup = simpson(a["CO2_ppm"].values, dx=diff, even='first') # proof that both methods have same answer:  area_sup_2 = area_s
                    
                    else:
                        raise ResidenceTimeMethodError
                        
                except ResidenceTimeMethodError as err:
                    print(err)
                    
                a.loc[[len(a)-1], "slope"] = slope
    
                # tail = a["CO2_ppm"][len(a)-1]/slope
                a_rest = a["CO2_ppm"].iloc[-1]/slope
                a_tot = area_sup + a_rest
                
                tau = a_tot/a["CO2_ppm"][0]
                a["tau_sec"] = tau
                
                try:     
                    if method=='iso':
                        # Taken from DIN ISO 16000-8:2008-12, Equation D2 units are cm3.m-3.sec
                        sa_num = ns_meas * (diff) * ((n - 1)/np.sqrt(n))
                        
                        # The uncertainty of the summed trapezoidal method itself is not covered by ISO 16000-8.
                        sa_tm = 0
                        
                    elif method=='trapez':
                        # Actually sa_num (the propagated uncertainty of the measurement) should be calculated this way
                        sa_num = (diff) * ns_meas * np.sqrt((2*n-1)/2*n) 
                        
                        # Aditionally the summed trapezoidal method itself has an uncertainty as well.
                        sa_tm = diff**2/12*(a["runtime"].loc[len(a)-1]-a["runtime"][0])*a["CO2_ppm"][0]/tau**2
                    
                    elif method=='simpson':
                        # Actually sa_num (the propagated uncertainty of the measurement) should be calculated this way
                        sa_num = 1/3*diff*ns_meas*np.sqrt(2+20*round(n/2-0.5))
                        
                        # Aditionally the summed trapezoidal method itself has an uncertainty as well.
                        sa_tm = diff**4/2880*(a["runtime"].loc[len(a)-1]-a["runtime"][0])*a["CO2_ppm"][0]/tau**4
                    
                    else:
                        raise ResidenceTimeMethodError
                        
                except ResidenceTimeMethodError as err:
                    print(err)
                
                s_lambda = a["slope"][:-1].std()/abs(a["slope"][:-1].mean())
                s_phi_e = a["slope"][:-1].std()/slope
        
                s_rest = np.sqrt(pow(s_lambda,2) + pow(s_phi_e,2))
                sa_rest = s_rest * a_rest
                s_area = np.sqrt(pow(sa_num,2) + pow(sa_tm,2) + pow(sa_rest,2))/a_tot
                s_total = np.sqrt(pow(s_area,2) + pow(s_rel_start,2))
                
                a.loc[:, "s_total"] = s_total*tau
                
                #%%%%% Calculate weighting factor 
                sup_exh_df = sup_exh_df.set_index('datetime')
                
                dfslice = sup_exh_df[a["datetime"][0]:a["datetime"][len(a)-1]]
                dfslice = dfslice.filter(['d calc exh-av', 'std d calc exh-av'])
                a = a.set_index('datetime')
                a = pd.concat([a, dfslice], axis = 1).reset_index()
                del dfslice
                
                from scipy.integrate import simpson
                area_weight = simpson(a["d calc exh-av"].values, dx=diff, even='first')
                # Actually sa_num (the propagated uncertainty of the measurement) should be calculated this way
                
                saw_num = 1/3*diff*np.mean(a["std d calc exh-av"])*np.sqrt(2+20*round(n/2-0.5))
                        
                # Aditionally the summed trapezoidal method itself has an uncertainty as well.
                saw_tm = diff**4/2880*(a["runtime"].loc[len(a)-1]-a["runtime"][0])*ddCmax_exhav["d calc exh-av"]
                
                saw = np.sqrt(pow(saw_num,2) + pow(saw_tm,2))
                
                area_weight = ufloat(area_weight, saw)
                weight = area_weight/(ufloat(ddCmax_exhav["d calc exh-av"],ddCmax_exhav["std d calc exh-av"])*a['runtime'].iloc[-1])
                
                a.loc[:, "weight"] = weight.n
                a.loc[:, "std weight"] = weight.s
                
                a.loc[:, "Cycle"] = s_cyc
    
                sup_exh_df.reset_index(inplace=True)
                
                #%%%%% Summarise
                df_tau_sup.append(a)
            
            else:
                if logging:
                    prRed("Since the supply cycle {} has a runtime of {} s it is outside [{}, {}]".format(s_cyc, lena, lb,  ub))
                pass
            
            s_cyc = s_cyc + 1
            
        else:
            pass
    #%%%% Supply tau from step-down curves 
    cyclnr_sup = []
    tau_list_sup = []
    stot_list_sup = []
    weight_list_sup = []
    saw_list_sup = []
    for jdf in df_tau_sup:
        cyclnr_sup.append(jdf["Cycle"][0])
        tau_list_sup.append(jdf["tau_sec"][0])
        stot_list_sup.append(jdf["s_total"][0])
        weight_list_sup.append(jdf["weight"][0])
        saw_list_sup.append(jdf["std weight"][0])
        
    df_tau_s = pd.DataFrame({'Cycle':cyclnr_sup,
                             'tau_sup':tau_list_sup, 
                             'std tau_sup':stot_list_sup, 
                             'weight':weight_list_sup, 
                             'std weight':saw_list_sup})
    
    # Filter outliers (see https://medium.com/@stevenewmanphotography/eliminating-outliers-in-python-with-z-scores-dd72ca5d4ead)
    df_tau_s['outliers'] = find_outliers(df_tau_s['tau_sup'])
    df_tau_s = df_tau_s[df_tau_s['outliers']==False]
    
    
    '''
    Weighting factor for the supply phases is not as important since the 
    residence times here are mostly normal distributed throughout the phases.
    Therefore it can be set low which means that almost all calcuated 
    residence times will be considered. The range for cfac_s is 0 to 1.
    Values >= 1 will autmatically trigger that the residence times with the 
    highest weighting factor will be chosen.
    '''
    cfac_s = 0.2
    df_tau_s2 = df_tau_s[df_tau_s['weight']>cfac_s]
    if len(df_tau_s2) == 0:
        df_tau_s2 = df_tau_s.nlargest(10, 'weight')
    
    tau_list_sup_u = unumpy.uarray(df_tau_s2['tau_sup'],df_tau_s2['std tau_sup'])
    weight_list_sup_u = unumpy.uarray(df_tau_s2['tau_sup'],df_tau_s2['std tau_sup']) 
    
    # Mean supply phase residence time
    tau_s_u = sum(tau_list_sup_u*weight_list_sup_u)/sum(weight_list_sup_u)
    
    
    # df_tau_s = pd.DataFrame({'nom' : [], 'std' : []})
    # count = 0
    # while (count < len(tau_list_sup_u)):
    #     df_tau_s.loc[count,['nom']] = tau_list_sup_u[count].n
    #     df_tau_s.loc[count,['std']] = tau_list_sup_u[count].s
    #     count = count + 1
        
    #%%%%% Plot: residence times of the step-down curves during supply-phase 
    if plot:
        import plotly.io as pio
    
        pio.renderers.default='browser'
        pd.options.plotting.backend = "matplotlib"
        #######################################################################
        pd.options.plotting.backend = "plotly"
        
        import plotly.io as pio
        
        pio.renderers.default='browser'
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        # Create figure with secondary y-axis
        fig2 = make_subplots(specs=[[{"secondary_y": True}]])
                
        fig2.add_trace(go.Scatter(name='Verweilzeit',
                                 x = df_tau_s['Cycle'], 
                                 y = df_tau_s['tau_sup'],
                                 error_y=dict(value=df_tau_s['std tau_sup'].max())
                                 ),
                       secondary_y=False,
                       )
        
        fig2.add_trace(go.Scatter(name='Gewichtung',
                                 x = df_tau_s['Cycle'], 
                                 y = df_tau_s['weight'],
                                 error_y=dict(value=df_tau_s['std weight'].max())
                                 ),
                       secondary_y=True,
                       )
        
        fig2.update_layout(
            title="Zuluft",
            xaxis_title="Zyklusnummer",
            yaxis_title=r'Verweilzeit $\bar{t}_1$',
            legend_title="Legende",
            font=dict(
                family="Segoe UI",
                size=18,
                color="black"
            )
        )
        
        fig2.show()
        
        import plotly.io as pio
        
        pio.renderers.default='browser'
        pd.options.plotting.backend = "matplotlib"
    
    #%% Marking dataframes exhaust
    """Marks every exhaust dataframe with a number for later anaysis """
    
    n = 1
    df_exh3 = df_exh2.copy().reset_index()
    
    
    mask = (df_exh3['datetime'] > start_date) & (df_exh3['datetime'] <= end_date)
    
    df_exh3 = df_exh3.loc[mask]
    
    
    for i,j in df_exh3.iterrows():
        try:
            # print(not pd.isnull(j["CO2_ppm"]), (np.isnan(df_exh3["CO2_ppm"][i+1])))
            if (not pd.isnull(j["CO2_ppm"])) and (np.isnan(df_exh3["CO2_ppm"][i+1])):
                df_exh3.loc[i,"num"] = n
                n = n+1
            elif (not pd.isnull(j["CO2_ppm"])):
                df_exh3.loc[i,"num"] = n
        except KeyError:
            pass
            # print("ignore the key error")
    
      
    #%%%% Exporrt a file with all the exhaust curves sorted in a matrix for an excel diagram    
    df_exh_list = []
    del dummy_df
    dummy_df = pd.DataFrame(columns=['datetime', 'CO2_ppm', 'num'])
    for i in range(1, int(df_exh3.num.max()+1)):

        try:
            if export_sublist and len(df_sup3.loc[df_exh3["num"]==i]) > 3:
                dummy_df = dummy_df.append(df_exh3.loc[df_exh3["num"]==(i)])
                dummy_df = dummy_df.rename(columns = {'CO2_ppm':'CO2_ppm_{}'.format(i)})
            
            
        except KeyError:
            pass
            # print("ignore the key error")

        df_exh_list.append(df_exh3.loc[df_exh3["num"]==i])
    
    del dummy_df["num"]
    if logging:
        dummy_df.to_csv(r'D:\Users\sauerswa\wichtige Ordner\sauerswa\Codes\Python\Recirculation\export\df_exh_{}_{}.csv'.format(database, aperture_sensor), index=True) 
        
        
    #%%% Exhaust tau
    # this method can be replicated in Excel for crossverification
   
    
    #%%%% Calculates tau based in area under the curve
    df_tau_exh = []
    e_cyc = 0
    for e in df_exh_list:
        if len(e) > 3:
            
            # %%%%% Structure columns
            
            b = e.reset_index(drop = True)                                    # Overwride the dummy dataframe "a" by the currently chosen supply decay curve.
            b['CO2_ppm_reg'] = b.eval(res.loc[0, "equation"])                   # See: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.eval.html?highlight=pandas%20dataframe%20eval#pandas.DataFrame.eval    
            b = b.rename(columns = {'CO2_ppm':'CO2_ppm_original', 'CO2_ppm_reg': 'CO2_ppm'})
            b = b.drop_duplicates(subset=['datetime'])
            b = b.loc[:, ["datetime", "CO2_ppm_original", "CO2_ppm"]]
            b = b.dropna()
            
            diff = (b["datetime"][1] - b["datetime"][0]).seconds
            b["runtime"] = np.arange(0,len(b) * diff, diff)
            
            lenb = b["runtime"].iloc[-1]            
            
            if b["runtime"].iloc[-1]>T/2*(1-filter_maxTrel) and b["runtime"].iloc[-1]<T/2*(1+filter_maxTrel):
                
                if logging:
                    prYellow("Since: {} < {} < {}, I consider the exhaust cycle {}".format(lb, lenb, ub, e_cyc))
                
                # %%%%% Calculating the measurement uncertainty 
                '''
                    based on the uncertainties of the reference sensors and the 
                    deviation of the sensor during its calibration
                '''
                
                b["std CO2_ppm"] =  np.sqrt(np.square((b["CO2_ppm"] * accuracy2)) 
                                       + np.square(accuracy1) + np.square((b["CO2_ppm"] * accuracy4)) 
                                       + np.square(accuracy3) + np.square((b["CO2_ppm"] * accuracy6)) 
                                       + np.square(accuracy5)+ np.square(res.loc[0, "rse"]))
                
                #%%%%% Add mean concentrations indoor (V3) to the exhaust dataframe
                # '''
                #     mean concentrations: 
                #         mean concentration over all measurements from
                #         senors in the occupied space (V3)
                # '''
                
                # b = b.set_index('datetime')
                # index_list = b.index.tolist()
                
                # count = 0
                # while (count < len(index_list)):     
                #     index_list[count] = index_list[count].strftime("%Y-%m-%d %H:%M:%S")
                #     count = count + 1
                
                # b.loc[:,['mean_delta', 'std mean_delta']] = dfin_dCmean.loc[index_list]
                # b.reset_index(inplace=True)
                # b['delta_C_mean'] = b['mean_delta'][0] - b['CO2_ppm']
                # # b = b.rename(columns = {'CO2_ppm_original':'CO2_ppm_original', 
                # #                         'CO2_ppm': 'CO2_ppm_reg', 
                # #                         'mean_delta': 'indoor_mean', 
                # #                         'delta_C': 'CO2_ppm'})
                
                
                #%%%%% Add mean concentrations indoor (V3) to the exhaust dataframe
                # '''
                #     mean concentrations: 
                #         Based on the calculated spatial and statistical mean air
                #         age in the occupied space and the spacial average initial 
                #         concentration in the occupied space at t0.
                # '''
                
                # count = 0
                # b['room_av'] = pd.Series(dtype='float64')
                # b['std room_av'] = pd.Series(dtype='float64')
                # while (count < len(b['datetime'])):     
                #     value = mean_delta_0_room_u*unumpy.exp(-1/(alpha_mean_u)*((b['datetime'][count]-t0).total_seconds()/3600))
                #     b['room_av'][count] = value.n
                #     b['std room_av'][count] = value.s
                #     count = count + 1            
                
                #%%%%% Add exhaust concentrations indoor (V3) to the exhaust dataframe
                '''
                    exhaust concentrations: 
                        Based on the calculated spatial and statistical mean air
                        age in the occupied space and the spacial average initial 
                        concentration in the occupied space at t0.
                '''
                
                count = 0
                b['room_exh'] = pd.Series(dtype='float64')
                b['std room_exh'] = pd.Series(dtype='float64')
                while (count < len(b['datetime'])):     
                    value = mean_delta_0_room_u*umath.exp(-1/(2*alpha_mean_u)*\
                                    ((b['datetime'][count]-t0).total_seconds()/3600))
                    b['room_exh'][count] = value.n
                    b['std room_exh'][count] = value.s
                    count = count + 1
                
                #%%%%% Concentration level after infinit time
                '''
                    A step-up concentration curve approaches to a certain max
                    after infinit time. This concentration results out of the 
                    exhaust concentration of the room and therefore from the
                    residence time of V3.
                '''
                
                dC3e = sum(unumpy.uarray(b['room_exh'],b['std room_exh']))/\
                        len(unumpy.uarray(b['room_exh'],b['std room_exh']))
                
                #%%%%% Calculate Delta C between exhaust of V3 and and exhaust of V2
                '''
                    
                '''
                
                count = 0
                b['dC 2e3e exh'] = pd.Series(dtype='float64')
                b['std dC 2e3e exh'] = pd.Series(dtype='float64')
                while (count < len(b['datetime'])):     
                    dC_2e_u = ufloat(b["CO2_ppm"][count],b["std CO2_ppm"][count])
                    dC_3e_u = ufloat(b["room_exh"][count],b["std room_exh"][count])
                    value = dC_3e_u - dC_2e_u
                    b['dC 2e3e exh'][count] = value.n
                    b['std dC 2e3e exh'][count] = value.s
                    count = count + 1           
                
                #%%%%% Calculation of the logarithmic concentration curves
                
                b = b.dropna()
                b = b[b["dC 2e3e exh"] >= b["dC 2e3e exh"].iloc[0]*0.36]
                b["log"] = np.log(b["dC 2e3e exh"])
                b["std log"] = b['std dC 2e3e exh']/b["dC 2e3e exh"]
                b = b.dropna()
                
                #%%%%% Rename columns to usual nomenclature 
                
                b = b.rename(columns = {'CO2_ppm':'CO2_ppm_reg', 'dC 2e3e exh':'CO2_ppm', 'std dC 2e3e exh': 's_meas'})
                
                #%%%%% Start of integral calculation
                
                # ### Calculating the measurement uncertainty based on the uncertainties of the reference sensors and the deviation of the sensor during its calibration
                # b["s_meas"] =  np.sqrt(np.square((b["CO2_ppm"] * accuracy2)) 
                #                        + np.square(accuracy1) + np.square((a["CO2_ppm"] * accuracy4)) 
                #                        + np.square(accuracy3) + np.square((a["CO2_ppm"] * accuracy6)) 
                #                        + np.square(accuracy5)+ np.square(res.loc[0, "rse"]))
                ns_meas = b['s_meas'].mean()
                n = len(b['s_meas'])
                
                # the following parameters have already been set global
                # global sa_num, s_lambda, s_phi_e
                # global area_sup, s_rest, s_total, a_rest, a_tot,sa_num,s_lambda, s_phi_e,s_rest, sa_rest, s_area
                            
                ### ISO 16000-8 option to calculate slope (defined to be calculated by Spread-Sheat/Excel)
                b["t-te"] = b["runtime"] - b["runtime"].iloc[len(b)-1]
                
                b["lnte/t"] = b["log"].iloc[len(b)-1] - b["log"]                         # @DRK: The slope (as defined in ISO 16000-8) was always negative since the two subtrahend where in the wrong order.
                
                b["slope"] = b["lnte/t"] / b["t-te"]
                
                try:
                    if method=='iso':
                        slope = b["slope"].mean()
                        
                        sumconz = b["CO2_ppm"].iloc[1:-1].sum()
                        area_sup = (diff * (b["CO2_ppm"][0]/2 + sumconz + b["CO2_ppm"][len(b)-1]/2))
                        print('ATTENTION: ISO 16000-8 method has a weak uncertainty evaluation consider using trapezoidal method is correcting this.')
                        
                    elif method=='trapez':
                        ### More acurate option to calculate the solpe of each (sub-)curve
                        x1 = b["runtime"].values
                        y1 = b["log"].values
                        
                        from scipy.stats import linregress
                        slope = -linregress(x1,y1)[0]
                        
                        from numpy import trapz
                        area_sup = trapz(b["CO2_ppm"].values, dx=diff) # proof that both methods have same answer:  area_sup_2 = area_sup_1
                        print('ATTENTION: Trapezoidal method is used in ISO 16000-8 and here also considered in the uncertainty evaluation. However, more precise results are given by applying the Simpson-Rule.')
                    
                    elif method=='simpson':
                        ### More acurate option to calculate the solpe of each (sub-)curve
                        x1 = b["runtime"].values
                        y1 = b["log"].values
                        
                        from scipy.stats import linregress
                        slope = -linregress(x1,y1)[0]
                        
                        from scipy.integrate import simpson
                        area_sup = simpson(b["CO2_ppm"].values, dx=diff, even='first') # proof that both methods have same answer:  area_sup_2 = area_s
                    
                    else:
                        raise ResidenceTimeMethodError
                        
                except ResidenceTimeMethodError as err:
                    print(err)
                    
                b["slope"].iloc[len(b)-1] = slope
    
                # tail = a["CO2_ppm"][len(a)-1]/slope
                a_rest = b["CO2_ppm"].iloc[-1]/slope
                a_tot = area_sup + a_rest
                
                tau2 = a_tot/dC3e.n
                b["tau_sec"] = tau2
                
                try:    
                    if method=='iso':
                        # Taken from DIN ISO 16000-8:2008-12, Equation D2 units are cm3.m-3.sec
                        sa_num = ns_meas * (diff) * ((n - 1)/np.sqrt(n))
                        
                        # The uncertainty of the summed trapezoidal method itself is not covered by ISO 16000-8.
                        sa_tm = 0
                        
                    elif method=='trapez':
                        # Actually sa_num (the propagated uncertainty of the measurement) should be calculated this way
                        sa_num = (diff) * ns_meas * np.sqrt((2*n-1)/2*n) 
                        
                        # Aditionally the summed trapezoidal method itself has an uncertainty as well.
                        sa_tm = diff**2/12*(b["runtime"].iloc[len(b)-1]-b["runtime"].iloc[0])*b["CO2_ppm"].iloc[0]/tau2**2
                    
                    elif method=='simpson':
                        # Actually sa_num (the propagated uncertainty of the measurement) should be calculated this way
                        sa_num = 1/3*diff*ns_meas*np.sqrt(2+20*round(n/2-0.5))
                        
                        # Aditionally the summed trapezoidal method itself has an uncertainty as well.
                        sa_tm = diff**4/2880*(b["runtime"].iloc[len(b)-1]-b["runtime"].iloc[0])*b["CO2_ppm"].iloc[0]/tau2**4
                    
                    else:
                        raise ResidenceTimeMethodError
                        
                except ResidenceTimeMethodError as err:
                    print(err)
                
                s_lambda = b["slope"][:-1].std()/abs(b["slope"][:-1].mean())
                s_phi_e = b["slope"][:-1].std()/slope
        
                s_rest = np.sqrt(pow(s_lambda,2) + pow(s_phi_e,2))
                sa_rest = s_rest * a_rest
                s_area = np.sqrt(pow(sa_num,2) + pow(sa_tm,2) + pow(sa_rest,2))/a_tot
                # ATTENTION: s_total is a relative uncertainty!
                s_total = np.sqrt(pow(s_area,2) + pow(dC3e.s/dC3e.n,2))
                
                b.loc[:, "s_total"] = s_total*tau2
                
                #%%%%% Calculate weighting factor 
                sup_exh_df = sup_exh_df.set_index('datetime')
                
                dfslice = sup_exh_df[b["datetime"].iloc[0]:b["datetime"].iloc[len(b)-1]]
                dfslice = dfslice.filter(['d calc exh-av', 'std d calc exh-av'])
                b = b.set_index('datetime')
                b = pd.concat([b, dfslice], axis = 1).reset_index()
                del dfslice
                
                from scipy.integrate import simpson
                area_weight = simpson(b["d calc exh-av"].values, dx=diff, even='first')
                # Actually sa_num (the propagated uncertainty of the measurement) should be calculated this way
                
                saw_num = 1/3*diff*np.mean(b["std d calc exh-av"])*np.sqrt(2+20*round(n/2-0.5))
                        
                # Aditionally the summed trapezoidal method itself has an uncertainty as well.
                saw_tm = diff**4/2880*(b["runtime"].iloc[len(b)-1]-b["runtime"].iloc[0])*ddCmax_exhav["d calc exh-av"]
                
                saw = np.sqrt(pow(saw_num,2) + pow(saw_tm,2))
                
                area_weight = ufloat(area_weight, saw)
                weight = area_weight/(ufloat(ddCmax_exhav["d calc exh-av"],ddCmax_exhav["std d calc exh-av"])*b['runtime'].iloc[-1])
                
                b.loc[:, "weight"] = weight.n
                b.loc[:, "std weight"] = weight.s
                
                b.loc[:, "Cycle"] = e_cyc
    
                sup_exh_df.reset_index(inplace=True)
                
                #%%%%% Summarise
                df_tau_exh.append(b)
            else:
                if logging:
                    prRed("Since the exhaust cycle {} has a runtime of {} s it is outside [{}, {}]".format(e_cyc, lenb, lb,  ub))
                pass
            
            e_cyc = e_cyc + 1
        
        else:
            pass
    #%%%% Exhaust tau from step-up curves 
    cyclnr_exh = []
    tau_list_exh = []
    stot_list_exh = []
    weight_list_exh = []
    saw_list_exh = []
    for jdf in df_tau_exh:
        cyclnr_exh.append(jdf["Cycle"][0])
        tau_list_exh.append(jdf["tau_sec"][0])
        stot_list_exh.append(jdf["s_total"][0])
        weight_list_exh.append(jdf["weight"][0])
        saw_list_exh.append(jdf["std weight"][0])
        
    df_tau_e = pd.DataFrame({'Cycle':cyclnr_exh,
                             'tau_exh':tau_list_exh, 
                             'std tau_exh':stot_list_exh, 
                             'weight':weight_list_exh, 
                             'std weight':saw_list_exh})
    
    # Filter outliers (see https://medium.com/@stevenewmanphotography/eliminating-outliers-in-python-with-z-scores-dd72ca5d4ead)
    df_tau_e['outliers'] = find_outliers(df_tau_e['tau_exh'])
    df_tau_e = df_tau_e[df_tau_e['outliers']==False]
    
    '''
        From the plots later on one can clearly see that the residence time 
        of theexhaust phases increases with the number of cycles of the measu-
        rement. This is because over time there are less and lesser marked 
        fluid elements in the system with low residence times, since they have 
        already been washed out. The remaining marked (with tracer) elements are  
        those who have been stagnating or recycled till the current period. As a 
        consequence it is quite obvious, that the residence time of V2 will
        after infinit time approach to the residence time of V3.
        The actual mean residence time of V2 is neighter the one measured by
        the first period nor by the one after infinit time. 
        
        Since it is already known that the residence time of the exhaust phases
        will approach to the residence time of V3 the infinit concentration 
        where the step-up curves are approaching is the exhaust concentration
        of V3. However, as just realised the residence times increase from 
        one cycle to the next cycle. For the tracer measurements this means
        that the infinity concentration of a step-up curve should be at least 
        between the average concentration in the room and the calculated 
        exhaust concentration of V3. Substracting the average concentration 
        over time from the exhaust concentration over time gives a function
        which has one maximum. Around this maximum the driving concentration
        diverence between exhaust air and room average air should be maximal
        and therefore the distiction between exhaust air and average room air.
        
        A cfac_e close to 1 will select those calculated residence times form
        evaluated cicles around this maximum.
    '''
    cfac_e = 0.99
    df_tau_e2 = df_tau_e[df_tau_e['weight']>cfac_e]
    if len(df_tau_e2) == 0:
        df_tau_e2 = df_tau_e.nlargest(10, 'weight')
    
    tau_list_exh_u = unumpy.uarray(df_tau_e2['tau_exh'],df_tau_e2['std tau_exh'])
    weight_list_exh_u = unumpy.uarray(df_tau_e2['tau_exh'],df_tau_e2['std tau_exh']) 
    
    tau_e_u = sum(tau_list_exh_u*weight_list_exh_u)/sum(weight_list_exh_u)
    
    #%%%%% Plot: residence times of the step-up curves during exhaust-phase  
    if plot:
        import plotly.io as pio

        pio.renderers.default='browser'
        pd.options.plotting.backend = "matplotlib"
        #######################################################################
        pd.options.plotting.backend = "plotly"
        
        import plotly.io as pio
        
        pio.renderers.default='browser'
        import plotly.express as px
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        # Create figure with secondary y-axis
        fig = make_subplots(specs=[[{"secondary_y": True}]])
                       
        fig.add_trace(go.Scatter(name='Verweilzeit',
                                 x = df_tau_e['Cycle'], 
                                 y = df_tau_e['tau_exh'],
                                 error_y=dict(value=df_tau_e['std tau_exh'].max())
                                 ),
                       secondary_y=False,
                       )
        
        
        fig.add_trace(go.Scatter(name='Gewichtung',
                                 x = df_tau_e['Cycle'], 
                                 y = df_tau_e['weight'],
                                 error_y=dict(value=df_tau_e["std weight"].max())
                                 ),
                       secondary_y=True,
                       )
        
        fig.update_layout(
            title="Abluft",
            xaxis_title="Zyklusnummer",
            yaxis_title=r'Verweilzeit $\bar{t}_2$',
            legend_title="Legende",
            font=dict(
                family="Segoe UI",
                size=18,
                color="black"
            )
        )
        
        fig.show()
        
        import plotly.io as pio
        
        pio.renderers.default='browser'
        pd.options.plotting.backend = "matplotlib"
     
    #%%%% Calculating the period number expected delivering the expected mean residence time of V2
    '''
        From the plots above one can clearly see that the residence time of the
        exhaust phases increases with the number of cycles of the measurement.
        This is because over time there are less and lesser marked fluid 
        elements in the system with low residence times since they have 
        already been washed out. The remaining marked (with tracer) elements are  
        those who have been stagnating or recycled till the current period. As a 
        consequence it is quite obvious, that the residence time of V2 will
        after infinit time approach to the residence time of V3.
        The actual mean residence time of V2 is neighter the one measured by
        the first period nor by the one after infinit time. The "mean index"
        of the period representing the best value for the residence time of V2
        has to be calculated by a similar procedure as the residence times 
        themsleves.
        
        Since it is already known that the residence time of the exhaust phases
        will approach to the residence time of V3 the infinit concentration 
        where the step-up curves are approaching is the exhaust concentration
        of V3. However, as just realised the residence times increase from 
        one cycle to the next cycle. For the tracer measurements this means
        that the infinity concentration of a step-up curve should be at least 
        between the average concentration in the room and the calculated 
        exhaust concentration of V3. Substracting the average concentration 
        over time from the exhaust concentration over time gives a function
        which carries information about the size of the interval of possible 
        concentrations which fullfill this criterion
    '''
    
    # df_indexm_exh = []

    
    #%%%%% Calculate Delta tau between exhaust of tau_3 and and exhaust of tau_2
    # '''       
    # '''
    
    # count = 0
    # df_tau_e['dtau 2e3e exh'] = pd.Series(dtype='float64')
    # df_tau_e['std dtau 2e3e exh'] = pd.Series(dtype='float64')
    # while (count < len(b['datetime'])):     
    #     dtau_2e_u = ufloat(df_tau_e["nom"][count],df_tau_e["std"][count])
    #     value = 2*alpha_mean_u*3600 - dtau_2e_u
    #     df_tau_e['dtau 2e3e exh'][count] = value.n
    #     df_tau_e['std dtau 2e3e exh'][count] = value.s
    #     count = count + 1           
    
    # #%%%%% Calculation of the logarithmic concentration curves
    
    # df_tau_e["log"] = np.log(df_tau_e['dtau 2e3e exh'])
    # df_tau_e["std log"] = df_tau_e['std dtau 2e3e exh']/df_tau_e['dtau 2e3e exh']
    # df_tau_e = df_tau_e.dropna()
    
    # #%%%%% Start of integral calculation
    
    # diff = 1
    
    # ns_meas = df_tau_e['std dtau 2e3e exh'].mean()
    # n = len(df_tau_e['std dtau 2e3e exh'])
                
    # # Because the evaluation of the residence times t0 and t0 --> Will be considered differently
    # #df_tau_e['index'] = df_tau_e['index'] + fdelay

    # df_tau_e['index'] = np.arange(0,len(df_tau_e) * diff, diff)
    
    # ### ISO 16000-8 option to calculate slope (defined to be calculated by Spread-Sheat/Excel)
    # df_tau_e["i-ie"] = df_tau_e['index'] - df_tau_e['index'][len(df_tau_e)-1]
    
    # df_tau_e["lnie/i"] = df_tau_e["log"][len(df_tau_e)-1] - df_tau_e["log"]                         # @DRK: The slope (as defined in ISO 16000-8) was always negative since the two subtrahend where in the wrong order.
    
    # df_tau_e["slope"] = df_tau_e["lnie/i"] / df_tau_e["i-ie"]
    
    
    # ### More acurate option to calculate the solpe of each (sub-)curve
    # x1 = df_tau_e['index'].values
    # y1 = df_tau_e["log"].values
            
    # from scipy.stats import linregress
    # slope = -linregress(x1,y1)[0]
            
    # from scipy.integrate import simpson
    # area_sup = simpson(df_tau_e["dtau 2e3e exh"].values, dx=diff, even='first') # proof that both methods have same answer:  area_sup_2 = area_s
        
        
    # df_tau_e.loc[[len(b)-1], "slope"] = slope

    # # tail = a["CO2_ppm"][len(a)-1]/slope
    # a_rest = df_tau_e["dtau 2e3e exh"].iloc[-1]/slope
    # a_tot = area_sup + a_rest
    
    # indexm2 = a_tot/(2*alpha_mean_u.n*3600)
    # df_tau_e["indexm2"] = indexm2
    
    
    # # Actually sa_num (the propagated uncertainty of the measurement) should be calculated this way
    # sa_num = 1/3*diff*ns_meas*np.sqrt(2+20*round(n/2-0.5))
    
    # # Aditionally the summed trapezoidal method itself has an uncertainty as well.
    # sa_tm = diff**4/2880*(df_tau_e['index'].loc[len(df_tau_e)-1]-df_tau_e['index'][0])*df_tau_e["dtau 2e3e exh"][0]/indexm2**4
    
    
    # s_lambda = df_tau_e["slope"][:-1].std()/abs(df_tau_e["slope"][:-1].mean())
    # s_phi_e = df_tau_e["slope"][:-1].std()/slope

    # s_rest = np.sqrt(pow(s_lambda,2) + pow(s_phi_e,2))
    # sa_rest = s_rest * a_rest
    # s_area = np.sqrt(pow(sa_num,2) + pow(sa_tm,2) + pow(sa_rest,2))/a_tot
    # s_total = np.sqrt(pow(s_area,2) + pow(df_tau_e["std dtau 2e3e exh"][0]/df_tau_e["dtau 2e3e exh"][0],2))
    
    # df_tau_e.loc[:, "s_total"] = s_total

    # df_tau_exh.append(b)
    
    
    #%% returned values
    """
        Returns:
            t0 = initial timestamp of the start of the experiment
            tn = final timestamp of the evaluated data
            tau_e = exhaust residence time of the short-cut volume
            tau_s = exhaust residence time of the recirculation volume 
                    ("supply residence time")
    """

    return [database, experiment, aperture_sensor, t0, tn, 2*alpha_mean_u, 
            tau_e_u, df_tau_e, 
            tau_s_u, df_tau_s]

#%% Func: residence_Vflow_weighted
# def colect_local_Vflows_Resitimes(experiment = "W_I_e0_Herdern"):
#     experimentglo = CBO_ESHL(experiment)
    
#     dvdt = Summarise_vflows(experiment)
    
    
    
#     return vflow, resitime




def residence_Vflow_weighted(vflow = pd.DataFrame([[30, 60], [5, 10]], 
                                                  columns=['vol flow', 'std vol flow'], 
                                                  dtype=('float64')), 
                             resitime = pd.DataFrame([[64, 45], [5, 10]],
                                                     columns=['rtime', 'std rtime'], 
                                                     dtype=('float64'))
                             ):
    from uncertainties import unumpy
    
    try:
        if len(vflow) == len(resitime):
            resitime_u = unumpy.uarray(resitime['rtime'], resitime['std rtime'])
            vflow_u = unumpy.uarray(vflow['vol flow'],vflow['std vol flow']) 
            
            resitime_u = sum(resitime_u*vflow_u)/sum(vflow_u)
            
            resitime = pd.DataFrame(columns=['rtime', 'std rtime'], 
                                    dtype=('float64'))
            resitime = pd.DataFrame([{'rtime': resitime_u.n, 'std rtime': resitime_u.s}],dtype=('float64'))
        else:
            raise ValueError 
            pass
    except ValueError:
        string = prYellow('ValueError: The number of passed volume flows and residence times has to be equal.')
        return string
    
    return resitime

def Summarise_vflows(experiment = "W_I_e0_Herdern"):
    
    experimentglo = CBO_ESHL(experiment = experiment)
    dvdt = pd.DataFrame(columns=('experiment','volume_flow','volume_flow_std','level',
                                 'vdot_sup','vdot_sup_std','vdot_exh','vdot_exh_std'))
 
    try:
        if 'eshl' in experimentglo.database:
            try:
                if experimentglo.experiment[2] == 'I':
                    level = ['Kü_100', 'SZ01_100', 'SZ02_100', 'WZ_100']
                    for count in range(len(level)):     
                        dvdt = dvdt.append(experimentglo.volume_flow(level_eshl = level[count]), ignore_index=True)
                    try:
                        if experimentglo.experiment[4:6] == 'e0':
                            pass
                        else:
                            raise ValueError
                    except ValueError:
                        strin1 = prYellow('Only those cases with balanced volume flow settings are yet covered by Summarise_vflows().')
                        return strin1 
                    
                elif experimentglo.experiment[2] == 'H':
                    level = ['Kü_20', 'SZ01_20', 'SZ02_20', 'WZ_20']
                    for count in range(len(level)):     
                        dvdt = dvdt.append(experimentglo.volume_flow(level_eshl = level[count]), ignore_index=True)
                    try:
                        if experimentglo.experiment[4:6] == 'e0':
                            pass
                        else:
                            raise ValueError
                    except ValueError:
                        strin2 = prYellow('Only those cases with balanced volume flow settings are yet covered by Summarise_vflows().')
                        return strin2
                    pass
                else:
                    raise NameError
                    pass
            except NameError:
                strin3 = prYellow('CBO_ESHL.experiment has the wrong syntax. The 3rd string element must be "I" for "intensiv ventilation" or "H" for "humidity protection".')
                return strin3
            pass
        elif 'cbo' in experimentglo.database:
            try:
                if experimentglo.experiment[2] == 'I':
                    level = ['K1_St5', 'K2_St5', 'SZ_St5']
                    for count in range(len(level)):     
                        dvdt = dvdt.append(experimentglo.volume_flow(level_cbo = level[count]), ignore_index=True)
                    try:
                        if experimentglo.experiment[4:6] == 'e0':
                            pass
                        else:
                            raise ValueError
                    except ValueError:
                        string4 = prYellow('Only those cases with balanced volume flow settings are yet covered by Summarise_vflows().')
                        return string4
                elif experimentglo.experiment[2] == 'H':
                    level = ['K1_St4', 'K2_St4', 'SZ_St4']
                    for count in range(len(level)):     
                        dvdt = dvdt.append(experimentglo.volume_flow(level_cbo = level[count]), ignore_index=True)
                    try:
                        if experimentglo.experiment[4:6] == 'e0':
                            pass
                        else:
                            raise ValueError
                    except ValueError:
                        string5 = prYellow('ValueError: Only those cases with balanced volume flow settings are yet covered by Summarise_vflows().')
                        return string5
                    pass
                else:
                    raise NameError
                    pass
            except NameError:
                string6 = prYellow('NameError: CBO_ESHL.experiment has the wrong syntax. The 3rd string element must be "I" for "intensiv ventilation" or "H" for "humidity protection".')
                return string6
            pass
        else:
            raise NameError
            pass
    except NameError:
        string7 = prYellow('NameError: The current CBO_ESHL.database is not valid. Volumeflows can not be returned CBO_ESHL.volume_flow().')   
        return string7

    return dvdt

def Summarise_resitimes(experiment = "W_I_e0_Herdern"):
    
    experimentglo = CBO_ESHL(experiment = experiment)
        
    time = pd.read_sql_query("SELECT * FROM testdb.timeframes;", con = engine)      
    #standard syntax to fetch a table from Mysql; In this case a table with the 
    # short-names of the measurements, all the start and end times, the DB-name 
    # of the measurement and the required table-names of the DB/schema is loaded into a dataframe. 
    
    t = time["timeframes_id"][time.index[time['short_name'].isin([experiment])==True].tolist()[0]]-1
    
    table = time["tables"][t].split(",")                                        #Name of the ventilation device
    
    resitime = pd.DataFrame(index=range(len(table)),
                        columns=('Sensor',
                                 'av restime_3 in h','std av restime_3 in h',
                                 'av restime_2 in s','std av restime_2 in s',
                                 'av restime_1 in s','std av restime_1 in s'))
    
    for i in range(len(table)):
        df = residence_time_sup_exh(experiment=experiment, aperture_sensor = table[i], 
                                    periodtime=120, 
                                    experimentname=True, plot=False, 
                                    export_sublist=False, method='simpson',
                                    filter_maxTrel=0.25, logging=False)
        resitime.loc[i] = pd.Series({'Sensor':table[i], 
                                 'av restime_3 in h': df[5].n,'std av restime_3 in h': df[5].s,
                                 'av restime_2 in s': df[6].n,'std av restime_2 in s': df[6].s,
                                 'av restime_1 in s': df[8].n,'std av restime_1 in s': df[8].s})

    return resitime

def check_for_nan(numbers = {'set_of_numbers': [1,2,3,4,5,np.nan,6,7,np.nan,8,9,10,np.nan]}):
    import pandas as pd
    import numpy as np
    
    df = pd.DataFrame(numbers,columns=['set_of_numbers'])

    check_for_nan = df['set_of_numbers'].isnull().values.any()
    print (check_for_nan)
    
def summary_resitime_vflow(experiment = "W_I_e0_Herdern", reset=False):
    import pandas as pd
    import pickle as pk
    import os.path
    
    experimentglo = CBO_ESHL(experiment = experiment)
    
    try:
        if reset:
            with open(experiment + "_summary", "wb") as file_summary:
                summary = [Summarise_vflows(experiment = experiment), 
                           Summarise_resitimes(experiment = experiment)]
                pk.dump(summary, file_summary)
            pass
        elif os.path.exists(experiment + "_summary"):
            with open(experiment + "_summary", "rb") as file_summary:
                summary = pk.load(file_summary)
            pass
        else:
            with open(experiment + "_summary", "wb") as file_summary:
                summary = [Summarise_vflows(experiment = experiment), 
                           Summarise_resitimes(experiment = experiment)]
                pk.dump(summary, file_summary)
            string3 = 'No file "{}_summary" found. "summary" has been recreated and saved as "{}_summary".'.format(experiment, experiment) 
            pass
    except IOError:
        prYellow(string3)
    finally:
        file_summary.close()
    
    
     
    try:
        if os.path.exists(experiment + "_summary_final"):
            with open(experiment + "_summary_final", "rb") as file_summary:
                summary = pk.load(file_summary)
            pass
        else:
            with open(experiment + "_summary_final", "wb") as file_summary:
                try:
                    if experiment == (summary[0]['experiment'].loc[:]).all():
                        volume_flow = summary[0]['volume_flow'].loc[0]
                        std_volume_flow = summary[0]['volume_flow_std'].loc[0]
                        av_resitime_3_h = summary[1]['av restime_3 in h'].loc[0]
                        std_av_resitime_3_h = summary[1]['std av restime_3 in h'].loc[0]
                        del summary[0]['experiment'], summary[0]['volume_flow'], summary[0]['volume_flow_std']
                        del summary[1]['av restime_3 in h'], summary[1]['std av restime_3 in h']
                        summary[0] = summary[0].set_index('level')
                        summary[1] = summary[1].set_index('Sensor')
                        summary.insert(0, experiment)
                        summary.insert(1, pd.DataFrame([{'volume_flow': volume_flow, 
                                                         'std_volume_flow': std_volume_flow}]))
                        summary.insert(2, pd.DataFrame([{'av restime_3 in h': av_resitime_3_h, 
                                                         'std av restime_3 in h': std_av_resitime_3_h}]))
                        pass
                    else: 
                        string1 = 'ValueError: summary_resitime_vflow() received wrong data.' 
                        raise ValueError
                except ValueError:
                    prYellow(string1)
            
      
                try:
                    if 'eshl' in experimentglo.database:
                        relation = pd.DataFrame(data={'Level':['SZ01_100', 'SZ02_100', 'Kü_100', 'WZ_100','SZ01_20', 'SZ02_20', 'Kü_20', 'WZ_20'],
                                                      'Sensor': ['1l',     '2l',       '3l_kü', '3l_wz',   '1l',      '2l',     '3l_kü', '3l_wz']
                                                     })
                        pass
                    elif 'cbo' in experimentglo.database:
                        relation = pd.DataFrame(data={'Level':['K1_St4', 'K1_St4', 'K2_St4', 'SZ_St4', 'K1_St5', 'K1_St5', 'K2_St5', 'SZ_St5'],
                                                      'Sensor': ['1l','1l_sub','2l','3l','1l','1l_sub','2l', '3l']
                                                     })
                        pass
                    else:
                        string2 = 'NameError: The current CBO_ESHL.database is not valid. Volumeflows can not be returned CBO_ESHL.summary_resitime_vflow().'
                        raise NameError
                        pass
                except NameError:
                    prYellow(string2)
    
                relation = pd.MultiIndex.from_frame(relation)
                summary[3] = summary[3].reindex(index=relation, level=0)
                summary[4] = summary[4].reindex(index=relation, level=1)
                summary.insert(5, pd.concat([summary[3], summary[4]], 
                                            join="outer", axis=1))
                summary[5] = summary[5].dropna()   
                # del summary[3], summary[4]
                
                #%%% Local residence time dataframes
                supplyt = summary[5].loc[:,['av restime_1 in s', 'std av restime_1 in s']]
                supplyt = supplyt.reset_index()
                del supplyt['Level'], supplyt['Sensor']
                supplyt.rename(columns = {'av restime_1 in s':'rtime', 'std av restime_1 in s':'std rtime'}, inplace = True)
                
                exhaustt = summary[5].loc[:,['av restime_2 in s', 'std av restime_2 in s']]
                exhaustt = exhaustt.reset_index()
                del exhaustt['Level'], exhaustt['Sensor']
                exhaustt.rename(columns = {'av restime_2 in s':'rtime', 'std av restime_2 in s':'std rtime'}, inplace = True)
                
                #%%% Local volume flow dataframes
                supplyV = summary[5].loc[:,['vdot_sup', 'vdot_sup_std']]
                supplyV = supplyV.reset_index()
                del supplyV['Level'], supplyV['Sensor']
                supplyV.rename(columns = {'vdot_sup':'vol flow', 'vdot_sup_std':'std vol flow'}, inplace = True)
                
                exhuastV = summary[5].loc[:,['vdot_exh', 'vdot_exh_std']]
                exhuastV = exhuastV.reset_index()
                del exhuastV['Level'], exhuastV['Sensor']
                exhuastV.rename(columns = {'vdot_exh':'vol flow', 'vdot_exh_std':'std vol flow'}, inplace = True)
            
                #%%% Calculating the weighted residence times for the whole system
                summary.insert(6,residence_Vflow_weighted(supplyV, supplyt))
                summary[6].rename(columns = {'rtime':'av t1 in s', 'std rtime':'std av t1 in s'}, inplace = True)
                summary.insert(7,residence_Vflow_weighted(exhuastV, exhaustt))
                summary[7].rename(columns = {'rtime':'av t2 in s', 'std rtime':'std av t2 in s'}, inplace = True)

                pk.dump(summary, file_summary)
            
            string4 = 'No file "{}_summary_final" found. "summary" has been recreated and saved as "{}_summary_final".'.format(experiment, experiment) 
            pass
    except IOError:
        prYellow(string4)
    finally:
        file_summary.close()
    
    return  summary
    
"""
    Tasks to be done:
        1.) Include uncertainty evaluation for tau_e and tau_s to be returned 
            at the end as well
        2.) Include an option where the plots are turned off by default.
        3.) Only the plots "original [experiment]" and the final plot are
            interesting.

"""



experiments = ["S_H_e0_Herdern", "W_I_e0_Herdern","W_H_e0_Herdern","S_H_e0_ESHL","S_I_e0_ESHL"]

# summary = summary_resitime_vflow(experiment = "W_I_e0_Herdern")

summaryE = []

for e in experiments:
    summary = summary_resitime_vflow(e)
    summaryE.append(summary)



# S_H_e0_Herdern = [Summarise_vflows(experiment = "S_H_e0_Herdern"), 
#                   Summarise_resitimes(experiment = "S_H_e0_Herdern")
#                   ]

# W_I_e0_Herdern = [Summarise_vflows(experiment = "W_I_e0_Herdern"), 
#                   Summarise_resitimes(experiment = "W_I_e0_Herdern")
#                   ]

# Fehlerhaft
# S_I_e0_Herdern = [Summarise_vflows(experiment = "S_I_e0_Herdern"), 
#                   Summarise_resitimes(experiment = "S_I_e0_Herdern")
#                   ]

# W_H_e0_Herdern = [Summarise_vflows(experiment = "W_H_e0_Herdern"), 
#                   Summarise_resitimes(experiment = "W_H_e0_Herdern")
#                   ]

# S_H_e0_ESHL = [Summarise_vflows(experiment = "S_H_e0_ESHL"), 
#                   Summarise_resitimes(experiment = "S_H_e0_ESHL")
#                   ]
# S_I_e0_ESHL = [Summarise_vflows(experiment = "S_I_e0_ESHL"), 
#                   Summarise_resitimes(experiment = "S_I_e0_ESHL")
#                   ]

# Fehlerhaft
# W_H_e0_ESHL = [Summarise_vflows(experiment = "W_H_e0_ESHL"), 
#                   Summarise_resitimes(experiment = "W_H_e0_ESHL")
#                   ]

# Fehlerhaft
# W_I_e0_ESHL = [Summarise_vflows(experiment = "W_I_e0_ESHL"), 
#                   Summarise_resitimes(experiment = "W_I_e0_ESHL")
#                   ]



# dvdt1 = Summarise_vflows(experiment = "W_I_e0_Herdern")

# resitime1 = Summarise_resitimes(experiment = "W_I_e0_Herdern")

# restime = residence_Vflow_weighted(vflow = pd.DataFrame([[30, 60], [5, 10]], 
#                                                   columns=['vol flow', 'std vol flow'], 
#                                                   dtype=('float64')), 
#                               resitime = pd.DataFrame([[64, 45], [5, 10]],
#                                                       columns=['rtime', 'std rtime'], 
#                                                       dtype=('float64'))
#                               )

# a1 = residence_time_sup_exh(experiment='S_H_e0_Herdern',aperture_sensor = "2l", periodtime=120,
#                             experimentname=True, plot=True,
#                             export_sublist=False, method='simpson',
#                             filter_maxTrel=0.25, logging=False)
# a2 = residence_time_sup_exh(experiment='W_I_e0_Herdern',aperture_sensor = "2l", periodtime=120,
#                             experimentname=True, plot=True,
#                             export_sublist=False, method='simpson',
#                             filter_maxTrel=0.25, logging=False)
# a3 = residence_time_sup_exh(experimentno=16, deviceno=2, periodtime=120, 
#                            experimentname=True, plot=True, 
#                            export_sublist=False, method='simpson',
#                            filter_maxTrel=0.25, logging=False)

