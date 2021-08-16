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
    # from Outdoor_CO2 import outdoor # This function calculates the outdoor CO2 data
    experimentglo = CBO_ESHL(experiment = experiment, aperture_sensor = aperture_sensor)
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
    
    # database = time["database"][time.index[time['short_name'].isin([experiment])==True].tolist()[0]]    # Selects the name of the database as a string
    database = experimentglo.database
    
    #%%% Load background data
        
    
        
    #background, dummy = outdoor(str(t0), str(end), plot = False)                    # Syntax to call the background concentration function, "dummy" is only necessary since the function "outdoor" returns a tuple of a dataframe and a string.
    # background = background["CO2_ppm"].mean() 
    background = experimentglo.aussen()['meanCO2']                                  # Future: implement cyclewise background concentration; Till now it takes the mean outdoor concentration of the whole experiment.
    background_std = experimentglo.aussen()['sgm_CO2'] 
    
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
            elif not pd.isnull(j["CO2_ppm"]):
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
    
    # self.cdf1 = pd.read_sql_query("SELECT * FROM {}.{} WHERE datetime BETWEEN '{}' AND '{}'".format(self.database, self.table, self.t0, self.tn), con = self.engine) 
    #         self.cdf2 = self.cdf1.loc[:,["datetime", "CO2_ppm"]]
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
                                       + np.square(accuracy5)+ np.square(res.loc[0, "rse"])
                                       + np.square(background_std))
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

#%% residence_Vflow_weighted
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
            string = 'ValueError: The number of passed volume flows and residence times has to be equal.'
            raise ValueError 
            pass
    except ValueError:
        prYellow(string)
    
    return resitime

#%% Summarise_vflows
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
                            string1 = 'Only those cases with balanced volume flow settings are yet covered by Summarise_vflows().'
                            raise ValueError
                    except ValueError:
                        prYellow(string1)
 
                    
                elif experimentglo.experiment[2] == 'H':
                    level = ['Kü_20', 'SZ01_20', 'SZ02_20', 'WZ_20']
                    for count in range(len(level)):     
                        dvdt = dvdt.append(experimentglo.volume_flow(level_eshl = level[count]), ignore_index=True)
                    try:
                        if experimentglo.experiment[4:6] == 'e0':
                            pass
                        else:
                            string2 = 'Only those cases with balanced volume flow settings are yet covered by Summarise_vflows().'
                            raise ValueError
                    except ValueError:
                        prYellow(string2)
                    pass
                else:
                    string3 = 'CBO_ESHL.experiment has the wrong syntax. The 3rd string element must be "I" for "intensiv ventilation" or "H" for "humidity protection".'
                    raise NameError
                    pass
            except NameError:
                prYellow(string3)
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
                            string4 = 'Only those cases with balanced volume flow settings are yet covered by Summarise_vflows().'
                            raise ValueError
                    except ValueError:
                        prYellow(string4)
                elif experimentglo.experiment[2] == 'H':
                    level = ['K1_St4', 'K2_St4', 'SZ_St4']
                    for count in range(len(level)):     
                        dvdt = dvdt.append(experimentglo.volume_flow(level_cbo = level[count]), ignore_index=True)
                    try:
                        if experimentglo.experiment[4:6] == 'e0':
                            pass
                        else:
                            string5 = 'ValueError: Only those cases with balanced volume flow settings are yet covered by Summarise_vflows().'
                            raise ValueError
                    except ValueError:
                        prYellow(string5)
                    pass
                else:
                    string6 = 'NameError: CBO_ESHL.experiment has the wrong syntax. The 3rd string element must be "I" for "intensiv ventilation" or "H" for "humidity protection".'
                    raise NameError
                    pass
            except NameError:
                prYellow(string6)
            pass
        else:
            string7 = 'NameError: The current CBO_ESHL.database is not valid. Volumeflows can not be returned CBO_ESHL.volume_flow().'
            raise NameError
            pass
    except NameError:
        prYellow(string7)

    return dvdt

#%% Summarise_resitimes
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

#%% check_for_nan
def check_for_nan(numbers = {'set_of_numbers': [1,2,3,4,5,np.nan,6,7,np.nan,8,9,10,np.nan]}):
    import pandas as pd
    import numpy as np
    
    df = pd.DataFrame(numbers,columns=['set_of_numbers'])

    check_for_nan = df['set_of_numbers'].isnull().values.any()
    print (check_for_nan)

#%% summary_resitime_vflow    
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
                        summary.insert(1, experimentglo.volume())
                        summary.insert(2, pd.DataFrame([{'volume_flow': volume_flow, 
                                                         'std_volume_flow': std_volume_flow}]))
                        summary.insert(3, pd.DataFrame([{'av restime_3 in h': av_resitime_3_h, 
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
                summary[4] = summary[4].reindex(index=relation, level=0)
                summary[5] = summary[5].reindex(index=relation, level=1)
                summary.insert(6, pd.concat([summary[4], summary[5]], 
                                            join="outer", axis=1))
                summary[6] = summary[6].dropna()   
                # del summary[3], summary[4]
                
                #%%% Local residence time dataframes
                supplyt = summary[6].loc[:,['av restime_1 in s', 'std av restime_1 in s']]
                supplyt = supplyt.reset_index()
                del supplyt['Level'], supplyt['Sensor']
                supplyt.rename(columns = {'av restime_1 in s':'rtime', 'std av restime_1 in s':'std rtime'}, inplace = True)
                
                exhaustt = summary[6].loc[:,['av restime_2 in s', 'std av restime_2 in s']]
                exhaustt = exhaustt.reset_index()
                del exhaustt['Level'], exhaustt['Sensor']
                exhaustt.rename(columns = {'av restime_2 in s':'rtime', 'std av restime_2 in s':'std rtime'}, inplace = True)
                
                #%%% Local volume flow dataframes
                supplyV = summary[6].loc[:,['vdot_sup', 'vdot_sup_std']]
                supplyV = supplyV.reset_index()
                del supplyV['Level'], supplyV['Sensor']
                supplyV.rename(columns = {'vdot_sup':'vol flow', 'vdot_sup_std':'std vol flow'}, inplace = True)
                
                exhuastV = summary[6].loc[:,['vdot_exh', 'vdot_exh_std']]
                exhuastV = exhuastV.reset_index()
                del exhuastV['Level'], exhuastV['Sensor']
                exhuastV.rename(columns = {'vdot_exh':'vol flow', 'vdot_exh_std':'std vol flow'}, inplace = True)
            
                #%%% Calculating the weighted residence times for the whole system
                summary.insert(7,residence_Vflow_weighted(supplyV, supplyt))
                summary[7].rename(columns = {'rtime':'av t1 in s', 'std rtime':'std av t1 in s'}, inplace = True)
                summary.insert(8,residence_Vflow_weighted(exhuastV, exhaustt))
                summary[8].rename(columns = {'rtime':'av t2 in s', 'std rtime':'std av t2 in s'}, inplace = True)
                
                #%%% Calculating the short-cut volume V2
                tav2 = summary[8]['av t2 in s'].loc[0]                          # residence time short-cut volume, in s
                tav2_std = summary[8]['std av t2 in s'].loc[0]
                Vdt23 = summary[2]['volume_flow'].loc[0]                        # effective volume flow of the ventilation device, in m³/h
                Vdt23_std = summary[2]['std_volume_flow'].loc[0]
                V23 = summary[1]['Volume V23 in m³'].loc[0]                     # volume of the ventilated space, in m³
                V23_std = summary[1]['std Volume V23 in m³'].loc[0]
                alphaav3 = summary[3]['av restime_3 in h'].loc[0]/2             # average air age in the ventilated space, in h
                alphaav3_std = summary[3]['std av restime_3 in h'].loc[0]/2
                
                V2 = short_cut_volume(tav2 = tav2, tav2_std = tav2_std,            
                                     Vdt23 = Vdt23, Vdt23_std = Vdt23_std,          
                                     V23 = V23, V23_std = V23_std,             
                                     alphaav3 = alphaav3, alphaav3_std = alphaav3_std)
                
                summary[1] = pd.concat([summary[1], V2],join="outer", axis=1)
                
                #%%% Remaining volume V3 containing the occupied space
                V23 = summary[1]['Volume V23 in m³'].loc[0]                    # volume of the ventilated space, in m³
                V23_std = summary[1]['std Volume V23 in m³'].loc[0]
                V2 = summary[1]['short-cut volume V2 in m³'].loc[0]            # volume of the ventilated space, in m³
                V2_std = summary[1]['std short-cut volume V2 in m³'].loc[0]
                
                V3 = occupied_volume(V23, V23_std, V2, V2_std)
                
                summary[1] = pd.concat([summary[1], V3],join="outer", axis=1)
                
                #%%% Volume flow circulating through the volume of the occupied space
                V3 = summary[1]['occupied volume V3 in m³'].loc[0]             # volume of the ventilated space, in m³
                V3_std = summary[1]['std occupied volume V3 in m³'].loc[0]
                alphaav3                                                       # residence time of the occupied space, in h
                alphaav3_std
                
                Vdt3 = occupied_volumeflow(V3, V3_std, alphaav3, alphaav3_std)
                
                summary[2] = pd.concat([summary[2], Vdt3],join="outer", axis=1)
                
                #%%% Volume flow through the short-cut volume
                Vdt3 = summary[2]['volume flow Vdt3 in m³/h'].loc[0]             # volume of the ventilated space, in m³
                Vdt3_std = summary[2]['std volume flow Vdt3 in m³/h'].loc[0]
                Vdt23
                Vdt23_std
                
                Vdt2 = short_cut_volumeflow(Vdt3, Vdt3_std, Vdt23, Vdt23_std)
                
                summary[2] = pd.concat([summary[2], Vdt2],join="outer", axis=1)
                
                #%%% Short-cut (K) and stagnation (S) ratio
                Vdt23
                Vdt23_std
                Vdt2 = summary[2]['volume flow Vdt2 in m³/h'].loc[0]             # volume of the ventilated space, in m³
                Vdt2_std = summary[2]['std volume flow Vdt2 in m³/h'].loc[0]
                
                KS = short_cut_stagnation_ratio(Vdt23, Vdt23_std,Vdt2, Vdt2_std)
                '''
                KS = pd.DataFrame([{'Kurzschluss in %Vdt': K[0].n, 'std Kurzschluss in %Vdt': K[0].s,
                                    'Stagnation in %Vdt': S[0].n, 'std Stagnation in %Vdt': S[0].s }])
                '''
                summary.insert(9,KS)
                
                #%%% Nominal time constants tau3, tau2 and tau1
                '''
                The basic assumption of this model is that the distinction
                between the subsystems is idealy mixed behaviour of the
                subsystems. Meaning that the relative exchange efficiency is:
                
                                     tau          tau
                epsilon^(a,r) = -------------- = ----- = 50%
                                2 * alphaav       tav
                
                Therefore:
                    tau3 = alphaav
                    tau2 = 0.5 * tav2
                    tau1 = 0.5 * tav1                    
                '''                
                tau = pd.DataFrame([{'tau3 in h': alphaav3, 'std tau3 in h': alphaav3_std,
                                    'tau2 in h': 0.5*summary[8]['av t2 in s'].iloc[0]/3600, 'std tau2 in h': 0.5*summary[8]['std av t2 in s'].iloc[0]/3600,
                                    'tau1 in h': 0.5*summary[7]['av t1 in s'].iloc[0]/3600, 'std tau1 in h': 0.5*summary[7]['std av t1 in s'].iloc[0]/3600}])
                
                summary.insert(10,tau)
                
                #%%% Response characteristics of the recycle system 23
                RecycSys23 = recyclesystem(Vdt12=summary[2]['volume_flow'].loc[0], Vdt12_std=summary[2]['std_volume_flow'].loc[0],     # volume flow entering and leaving the whole system V12, in m³/h           
                                         Vdt2=summary[2]['volume flow Vdt3 in m³/h'].loc[0], Vdt2_std=summary[2]['std volume flow Vdt3 in m³/h'].loc[0],       # volume flow of the backfeed subsystem V2, in m³/h
                                         tau1=summary[10]['tau2 in h'].loc[0], tau1_std=summary[10]['std tau2 in h'].loc[0],   # nominal time constant of subsystem V1, in h
                                         tau2=summary[10]['tau3 in h'].loc[0], tau2_std=summary[10]['std tau3 in h'].loc[0])
                
                summary.insert(11,RecycSys23)
                
                #%%% Calculate exchange efficiency characteristics
                EE = exchange_efficiency(tau=summary[11]['tau in h'].loc[0],  tau_std=summary[11]['std tau in h'].loc[0],
                                         sigma_dimless=summary[11]['(sigma²)* in -'].loc[0],  sigma_dimless_std=summary[11]['std (sigma²)* in -'].loc[0])

                summary.insert(12,EE)
                
                #%%% Calculate recirculation ratio
                R = recirculation_ratio(S=summary[9]['Stagnation in %Vdt'].iloc[0],S_std=summary[9]['std Stagnation in %Vdt'].loc[0],
                                        epsilonabs=summary[12]['epsilon in %t'].iloc[0],epsilonabs_std=summary[12]['std epsilon in %t'].iloc[0])
                
                summary[9] = pd.concat([summary[9], R],join="outer", axis=1)
                
                #%% Save a final summary file
                pk.dump(summary, file_summary)
            
            string4 = 'No file "{}_summary_final" found. "summary" has been recreated and saved as "{}_summary_final".'.format(experiment, experiment) 
            pass
    except IOError:
        prYellow(string4)
    finally:
        file_summary.close()
    
    return  summary

#%% exchange_efficiency
def exchange_efficiency(tau=0.0,  tau_std=99.9,
                        tav=0.0,  tav_std=9.99,
                        alphaav=0.0, alphaav_std=9.99,
                        n=0.0, n_std=9.99,
                        navex=0.0, navex_std=9.99,
                        navr=0.0, navr_std=9.99,
                        sigma_dimless=0.0,  sigma_dimless_std=9.99 # dimensonless variance
                        ):
    #%% Description
    '''
    epsilon^(a) can be calculated through various ways.
    
                    tau     navex       tau        1     navr       1
    epsilon^(a) := ----- = -------- = ---------- = --- * ------- = -------------  in %t
                    tav      n        2*alphaav     2       n       1 + sigma²*
         
         tau        nominal time constant,                  in h
         tav        average residence time,                 in h
         navex     exhaust exchange rate,                  in 1/h
         n          nominal exchange rate,                  in 1/h
         alphaav    average age in the system,              in h
         navr      average exchange rate in the system,    in 1/h
    '''
    
    #%% Neccessary packages
    from uncertainties import ufloat
    import pandas as pd
    import numpy as np
    import sympy as sym
    from sympy import init_printing
    from sympy import Symbol
    
    #%% Define parameters including uncertainty
    sigma_dimless = [ufloat(sigma_dimless, sigma_dimless_std), Symbol(r'\left(\sigma^2\right)^*')]
    tau = [ufloat(tau, tau_std), Symbol(r'\tau')]
    tav = [ufloat(tav, tav_std), Symbol(r'\bar{t}')]
    alphaav = [ufloat(alphaav, alphaav_std), Symbol(r'\left\langle\bar{\alpha}\right\rangle')]
    n = [ufloat(n, n_std), Symbol(r'n')]
    navex = [ufloat(navex, navex_std), Symbol(r'\left\langle\bar{n}\right\rangle_e')]
    navr = [ufloat(navr, navr_std), Symbol(r'\left\langle\bar{n}\right\rangle_r')]
    
    #%% Equation
    epsilon = [None]*2
 
    for i in range(2):
    
        if tau[0].n>0:
            n[0] = 1/tau[0]
            n[1] = 1/tau[1]
        elif n[0].n>0: 
            tau[0] = 1/n[0]
            tau[1] = 1/n[1]
        else:
            pass
        
        if tav[0].n>0:
            alphaav[0] = tav[0]*0.5
            alphaav[1] = tav[1]*0.5
            
            navex[0] = 1/tav[0]
            navex[1] = 1/tav[1]
            
            navr[0] = 1/alphaav[0]
            navr[1] = 1/alphaav[1]
        elif alphaav[0].n>0: 
            tav[0] = 2*alphaav[0]
            tav[1] = 2*alphaav[1]
            
            navex[0] = 1/tav[0]
            navex[1] = 1/tav[1]
            
            navr[0] = 1/alphaav[0]
            navr[1] = 1/alphaav[1]
        elif navex[0].n>0:
            tav[0] = 1/navex[0]
            tav[1] = 1/navex[1]
            
            alphaav[0] = tav[0]*0.5
            alphaav[1] = tav[1]*0.5
            
            navr[0] = 1/alphaav[0]
            navr[1] = 1/alphaav[1]
        elif navr[0].n>0:
            alphaav[0] = 1/navr[0]
            alphaav[1] = 1/navr[1]
        else:
            pass
    
        if tau[0].n>0 and tav[0].n>0:
            epsilon[0] = tau[0]/tav[0]
            epsilon[1] = tau[1]/tav[1]
            
            sigma_dimless[0] = 1/epsilon[0]-1
            sigma_dimless[1] = 1/epsilon[1]-1
            pass
        elif sigma_dimless[0].n>0 and tav[0].n>0:
            epsilon[0] = 1/(1+sigma_dimless[0])
            epsilon[1] = 1/(1+sigma_dimless[1])
            
            tau[0] = tav[0]/(1+sigma_dimless[0])
            tau[1] = tav[1]/(1+sigma_dimless[1])
            pass
        elif sigma_dimless[0].n>0 and tau[0].n>0:
            epsilon[0] = 1/(1+sigma_dimless[0])
            epsilon[1] = 1/(1+sigma_dimless[1])
            
            tav[0] = tau[0]*(1+sigma_dimless[0])
            tav[1] = tau[1]*(1+sigma_dimless[1])
            pass
        else:
            pass

    #%% Summarise in a dataframe
    global df_EE
    df_EE = pd.DataFrame([{'epsilon in %t':epsilon[0].n*100, 'std epsilon in %t':epsilon[0].s*100,
                           'tau in h':tau[0].n, 'std tau in h':tau[0].s,
                           'tav in h':tav[0].n, 'std tav in h':tav[0].s,
                           'alphaav in h':alphaav[0].n, 'std alphaav in h':alphaav[0].s,
                           'n in 1/h':n[0].n, 'std n in 1/h':n[0].s,
                           'navex in 1/h':navex[0].n, 'std navex in 1/h':navex[0].s,
                           'navr in 1/h':navr[0].n, 'std navr in 1/h':navr[0].s,
                           '(sigma²)* in -': sigma_dimless[0].n, 'std (sigma²)* in -': sigma_dimless[0].s
                           }])

    
    try:
        if df_EE.isin([0]).any().any(): 
            if sigma_dimless[0].n>0:
                epsilon[0] = 1/(1+sigma_dimless[0])
                epsilon[1] = 1/(1+sigma_dimless[1])
                df_EE = pd.DataFrame([{'(sigma²)* in -': sigma_dimless[0].n, 'std (sigma²)* in -': sigma_dimless[0].s}])
                print('ATTENTION: I can only calculate the exchange efficiency. Please, pass another argument.')
            else:  
                pass
            string = prYellow('ValueError: exchange_efficiency() misses a value.')
            raise ValueError
    except ValueError:
        string
 
    return df_EE



#%% short_cut_volume
def short_cut_volume(tav2, tav2_std,            # residence time short-cut volume, in s 
                     Vdt23, Vdt23_std,          # effective volume flow of the ventilation device, in m³/h
                     V23, V23_std,              # volume of the ventilated space, in m³
                     alphaav3, alphaav3_std,   # average air age in the ventilated space, in h
                     printeq = False,
                     calc=True):
    #%% Description
    '''
    This function is only applicable for a recirculation system according to:
    Nauman, E. B., Buffham, B. A. (1983), Mixing in continuous flow systems. Wiley, New York, ISBN: 978-0471861911, page: 21.
    
    Assumption:
        - Recirculation between two subsystems of a system to be calculated
        - subsystems are fully mixed -> nominal time constant of the subsystems equals their mean air age
        - comparing ages and time constants between the subsystems they are usually not equal 
    '''
    
    #%% Neccessary packages
    from uncertainties import ufloat
    
    from sympy import init_printing
    from sympy import symbols
    from sympy import Eq
    from sympy import Rational
    from sympy import Add
    
    #%% Define parameters including uncertainty
    tav2 = [ufloat(tav2, tav2_std), symbols(r'\left\langle\bar{t}\right\rangle_\mathrm{2}')]
    Vdt23 = [ufloat(Vdt23, Vdt23_std), symbols(r'{\dot{V}}_\mathrm{23}')]
    V23 = [ufloat(V23, V23_std), symbols(r'V_\mathrm{23}')]
    alphaav3 = [ufloat(alphaav3, alphaav3_std), symbols(r'\left\langle\bar{\alpha}\right\rangle_\mathrm{3}')]
    stoh = [3600, symbols(r'3600~\frac{\mathrm{s}}{\mathrm{h}}')]
    
    #%% Equation
    V2 = [None]*2
    
    try:    
        if printeq:
            init_printing()
            Eq(V2[1], Rational( Add(Rational(tav2[1],stoh[1]),(Vdt23[1]+V23[1]/alphaav3[1])),2+Rational(tav2[1],alphaav3[1])))
        elif printeq and calc:
            V2[0] = (tav2[0]/stoh[0]*(Vdt23[0]+V23[0]/alphaav3[0]))/(2+tav2[0]/alphaav3[0])
            init_printing()
            Eq(V2[1], Rational( Add(Rational(tav2[1],stoh[1]),(Vdt23[1]+V23[1]/alphaav3[1])),2+Rational(tav2[1],alphaav3[1])))
        else:
            V2[0] = (tav2[0]/stoh[0]*(Vdt23[0]+V23[0]/alphaav3[0]))/(2+tav2[0]/alphaav3[0])
    except NameError:
        V2[0] = (tav2[0]/stoh[0]*(Vdt23[0]+V23[0]/alphaav3[0]))/(2+tav2[0]/alphaav3[0])
    
    #%% Summarise in a dataframe
    V2 = pd.DataFrame([{'short-cut volume V2 in m³': V2[0].n, 'std short-cut volume V2 in m³': V2[0].s}])

    return V2

#%%occupied_volume
def occupied_volume(V23, V23_std,              # volume of the ventilated space, in m³
                    V2, V2_std,                # total short-cut volume (sum of all short-cut volumes around the indoor aperatures), in m³
                    printeq = False,
                    calc=True):
    #%% Description
    '''
    Volume remaining for the occupied space, in m³
    '''
    
    #%% Neccessary packages
    from uncertainties import ufloat
    
    from sympy import init_printing
    from sympy import symbols
    from sympy import Eq
    
    #%% Define parameters including uncertainty
    V23 = [ufloat(V23, V23_std), symbols(r'V_\mathrm{23}')]
    V2 = [ufloat(V2, V2_std), symbols(r'V_\mathrm{2}')]
    
    #%% Equation
    V3 = [None]*2
    
    try:    
        if printeq:
            init_printing()
            Eq(V3[1], V23[1] - V2[1])
        elif printeq and calc:
            V3[0] = V23[0] - V2[0]
            init_printing()
            Eq(V3[1], V23[1] - V2[1])
        else:
            V3[0] = V23[0] - V2[0]
    except NameError:
        V3[0] = V23[0] - V2[0]
    
    #%% Summarise in a dataframe
    V3 = pd.DataFrame([{'occupied volume V3 in m³': V3[0].n, 'std occupied volume V3 in m³': V3[0].s}])

    return V3

#%% volumeflow stagnating (avialable for the occupied zone)
def occupied_volumeflow(V3, V3_std,              # volume of the ventilated space, in m³
                        alphaav3, alphaav3_std,                # total short-cut volume (sum of all short-cut volumes around the indoor aperatures), in m³
                        printeq = False,
                        calc=True):
    #%% Description
    '''
    The volume for the occupied space is ideally mixed therefore its nominal
    time constant tau3 equals its average age alphaav3.
    '''
    
    #%% Neccessary packages
    from uncertainties import ufloat
    
    from sympy import init_printing
    from sympy import symbols
    from sympy import Eq
    from sympy import Rational
    
    #%% Define parameters including uncertainty
    V3 = [ufloat(V3, V3_std), symbols(r'V_\mathrm{3}')]
    alphaav3 = [ufloat(alphaav3, alphaav3_std), symbols(r'\left\langle\bar{\alpha}\right\rangle_\mathrm{3}')]
    
    #%% Equation
    Vdt3 = [None]*2
    
    try:    
        if printeq:
            init_printing()
            Eq(Vdt3[1], Rational(V3[1],alphaav3[1]))
        elif printeq and calc:
            Vdt3[0] = V3[0]/alphaav3[0]
            init_printing()
            Eq(Vdt3[1], Rational(V3[1],alphaav3[1]))
        else:
            Vdt3[0] = V3[0]/alphaav3[0]
    except NameError:
        Vdt3[0] = V3[0]/alphaav3[0]
    
    #%% Summarise in a dataframe
    Vdt3 = pd.DataFrame([{'volume flow Vdt3 in m³/h': Vdt3[0].n, 'std volume flow Vdt3 in m³/h': Vdt3[0].s}])

    return Vdt3

#%% short-cut volumeflow
def short_cut_volumeflow(Vdt3, Vdt3_std,                # volume flow for the occupied space, in m³/h
                         Vdt23, Vdt23_std,            # volume flow of the ventilated space, in m³/h
                         printeq = False,
                         calc=True):
    #%% Description
    '''
    According to Nauman et al. the short-cut volume flow of a back-feed 
    recirculation system is the sum out of the system volume flow V23 and 
    the recycle volume flow V3
    
    (The volume for the short-cut volume is ideally mixed therefore its nominal
    time constant tau2 equals its average age alphaav2.)
    '''
    
    #%% Neccessary packages
    from uncertainties import ufloat
    
    from sympy import init_printing
    from sympy import symbols
    from sympy import Eq
    
    #%% Define parameters including uncertainty
    Vdt3 = [ufloat(Vdt3, Vdt3_std), symbols(r'\dot{V}_\mathrm{3}')]
    Vdt23 = [ufloat(Vdt23, Vdt23_std), symbols(r'\dot{V}_\mathrm{23}')]
    
    #%% Equation
    Vdt2 = [None]*2
    
    try:    
        if printeq:
            init_printing()
            Eq(Vdt2[1], Vdt3[1] + Vdt23[1])
        elif printeq and calc:
            Vdt2[0] = Vdt3[0] + Vdt23[0]
            init_printing()
            Eq(Vdt2[1], Vdt3[1] + Vdt23[1])
        else:
            Vdt2[0] = Vdt3[0] + Vdt23[0]
    except NameError:
        Vdt2[0] = Vdt3[0] + Vdt23[0]
    
    #%% Summarise in a dataframe
    Vdt2 = pd.DataFrame([{'volume flow Vdt2 in m³/h': Vdt2[0].n, 'std volume flow Vdt2 in m³/h': Vdt2[0].s}])

    return Vdt2

#%% short-cut and stagnation ratio
def short_cut_stagnation_ratio(Vdt23, Vdt23_std,        # volume flow of the ventilated space, in m³/h            
                               Vdt2, Vdt2_std,          # volume flow of the short-cut volume, in m³/h
                               printeq = False,
                               calc=True):
    #%% Description
    '''
    K and S rate the the volume flows moving inside the building zone.
    
             Vdt23       Vdt23        Vdt3
    K := ------------- = ----- = 1 - ------ =: 1 - S
         Vdt23 + Vdt3    Vdt2         Vdt2
         
         K      Short-cut rate (german: Kurzschlussrate),       in %Vdt
         S      Stagnation rate (german: Stagnationsrate),      in %Vdt
         Vdt23  Volume flow entering and leaving the system,    in m³/h
         Vdt2   Volume flow through the short-cut volume,       in m³/h
         Vdt3   Volume flow through the rest of the room,       in m³/h
    '''
    
    #%% Neccessary packages
    from uncertainties import ufloat
    
    from sympy import init_printing
    from sympy import Symbol
    from sympy import Eq
    from sympy import Rational
    
    #%% Define parameters including uncertainty
    Vdt2 = [ufloat(Vdt2, Vdt2_std), Symbol(r'\dot{V}_\mathrm{2}')]
    Vdt23 = [ufloat(Vdt23, Vdt23_std), Symbol(r'\dot{V}_\mathrm{23}')]
    
    #%% Equation
    K = [None]*2
    S = [None]*2
    
    try:    
        if printeq:
            init_printing()
            Eq(K[1], Rational(Vdt23[1], Vdt2[1]))
        elif printeq and calc:
            K[0] = Vdt23[0]/Vdt2[0]*100
            S[0] = 100 - K[0]
            init_printing()
            Eq(K[1], Rational(Vdt23[1], Vdt2[1]))
        else:
            K[0] = Vdt23[0]/Vdt2[0]*100
            S[0] = 100 - K[0]
    except NameError:
        K[0] = Vdt23[0]/Vdt2[0]*100
        S[0] = 100 - K[0]
    
    #%% Summarise in a dataframe
    KS = pd.DataFrame([{'Kurzschluss in %Vdt': K[0].n, 'std Kurzschluss in %Vdt': K[0].s,
                        'Stagnation in %Vdt': S[0].n, 'std Stagnation in %Vdt': S[0].s }])

    return KS

#%% occupied_volumeflow
def short_cut_stagnation_ratio(Vdt23, Vdt23_std,        # volume flow of the ventilated space, in m³/h            
                               Vdt2, Vdt2_std,          # volume flow of the short-cut volume, in m³/h
                               printeq = False,
                               calc=True):
    #%% Description
    '''
    K and S rate the the volume flows moving inside the building zone.
    
             Vdt23       Vdt23        Vdt3
    K := ------------- = ----- = 1 - ------ =: 1 - S
         Vdt23 + Vdt3    Vdt2         Vdt2
         
         K      Short-cut rate (german: Kurzschlussrate),       in %Vdt
         S      Stagnation rate (german: Stagnationsrate),      in %Vdt
         Vdt23  Volume flow entering and leaving the system,    in m³/h
         Vdt2   Volume flow through the short-cut volume,       in m³/h
         Vdt3   Volume flow through the rest of the room,       in m³/h
    '''
    
    #%% Neccessary packages
    from uncertainties import ufloat
    
    from sympy import init_printing
    from sympy import Symbol
    from sympy import Eq
    from sympy import Rational
    
    #%% Define parameters including uncertainty
    Vdt2 = [ufloat(Vdt2, Vdt2_std), Symbol(r'\dot{V}_\mathrm{2}')]
    Vdt23 = [ufloat(Vdt23, Vdt23_std), Symbol(r'\dot{V}_\mathrm{23}')]
    
    #%% Equation
    K = [None]*2
    S = [None]*2
    
    try:    
        if printeq:
            init_printing()
            Eq(K[1], Rational(Vdt23[1], Vdt2[1]))
        elif printeq and calc:
            K[0] = Vdt23[0]/Vdt2[0]*100
            S[0] = 100 - K[0]
            init_printing()
            Eq(K[1], Rational(Vdt23[1], Vdt2[1]))
        else:
            K[0] = Vdt23[0]/Vdt2[0]*100
            S[0] = 100 - K[0]
    except NameError:
        K[0] = Vdt23[0]/Vdt2[0]*100
        S[0] = 100 - K[0]
    
    #%% Summarise in a dataframe
    KS = pd.DataFrame([{'Kurzschluss in %Vdt': K[0].n, 'std Kurzschluss in %Vdt': K[0].s,
                        'Stagnation in %Vdt': S[0].n, 'std Stagnation in %Vdt': S[0].s }])

    return KS

#%% occupied_volumeflow
def recyclesystem(Vdt12=40.0, Vdt12_std=5.0,     # volume flow entering and leaving the whole system V12, in m³/h           
                  Vdt2=50.0, Vdt2_std=6.0,       # volume flow of the backfeed subsystem V2, in m³/h
                  tau1=0.009, tau1_std=0.0001,   # nominal time constant of subsystem V1, in h
                  tau2=3.2, tau2_std=0.01):       # nominal time constant of subsystem V2, in h

    #%% Description
    '''
    This function is only applicable for a recirculation system according to:
    Nauman, E. B., Buffham, B. A. (1983), Mixing in continuous flow systems. Wiley, New York, ISBN: 978-0471861911, page: 21.
    
    Assumption:
        - Recirculation between two subsystems of a system to be calculated
        - subsystems are fully mixed -> nominal time constant of the subsystems equals their mean air age
        - comparing ages and time constants between the subsystems they are usually not equal 
    '''
    
    #%% Neccessary packages
    from uncertainties import ufloat
    import sympy as sym
    import numpy as np
    # from sympy import init_printing
    # from sympy import symbols
    # # from sympy import Eq
    # # from sympy import Rational
    
    #%% Define parameters including uncertainty
    Vdt12 = [ufloat(Vdt12, Vdt12_std), sym.Symbol(r'\dot{V}_{12}')]             # volume flow entering and leaving the whole system V12, in m³/h 
    Vdt2 = [ufloat(Vdt2, Vdt2_std), sym.Symbol(r'\dot{V}_{2}')]                 # volume flow of the backfeed subsystem V2, in m³/h
    tau1 = [ufloat(tau1, tau1_std), sym.Symbol('tau1')]                         # nominal time constant of subsystem V1, in h
    tau2 = [ufloat(tau2, tau2_std), sym.Symbol('tau2')]                         # nominal time constant of subsystem V2, in h
    
    s = sym.Symbol('s')                                                         # frequency parameter of a Laplace-transform, in 1/h
        
    #%% Equation
    tau12 = [None]*4
    variance12 = [None]*4
    skew12 = [None]*4
    kurtosis12 = [None]*4
    
    f1 = 1/(tau1[1]*s+1)                                                           # Respones function of subsystem 1
    f2 = 1/(tau2[1]*s+1)                                                           # Respones function of subsystem 2
    f12 = f1/(1+Vdt2[1]/Vdt12[1]*(1-f1*f2))                                           # Respones function of system 12
    f12ln = sym.ln(f12)
    
    #%%% Sympy expressions
    #%%%% 1st kumulant = 1st moment of the propability distribution = tau12
    d1f12lnds1 = f12ln.diff(s)
    tau12[1] = np.power(-1,1)*d1f12lnds1.subs({s:0})                               # mean value, tau in h
    
    #%%%% 2nd kumulant = 1st central moment of the propability distribution = variance12
    d2f12lnds2 = d1f12lnds1.diff(s)
    variance12[1] = np.power(-1,2)*d2f12lnds2.subs({s:0})                          # variance, sigma in h^2
    variance12[3] = variance12[1]/np.power(tau12[1],2)                                 # dimensionless variance, sigma* in -
    
    #%%%% 3rd kumulant = 2nd central moment of the propability distribution = skew12
    d3f12lnds3 = d2f12lnds2.diff(s)
    skew12[1] = np.power(-1,3)*d3f12lnds3.subs({s:0})                              # skewness, mu3 in h^3
    skew12[3] = skew12[1]/np.power(tau12[1],3)                                         # dimensionless skewness, mu3* in -
    
    #%%%% 4th kumulant lead to 3rd central moment of the propability distribution = kurtosis12
    d4f12lnds4 = d3f12lnds3.diff(s)
    kurtosis12[1] = np.power(-1,4)*d4f12lnds4.subs({s:0})+3*np.power(variance12[1],2) # kurtosis, mu4 in h^4
    kurtosis12[3] = kurtosis12[1]/np.power(tau12[1],4)                                 # dimensionless kurtosis, mu4* in -
    
    #%%% Calculations including uncertainties
    #%%%% 1st kumulant = 1st moment of the propability distribution = tau12
    tau12[0] = tau1[0]+(Vdt2[0]*(tau1[0]+tau2[0]))/Vdt12[0]
    
    #%%%% 2nd kumulant = 2nd central moment of the propability distribution = variance12
    a = 2*np.power(tau1[0],2)
    b = + tau1[0]*(-tau1[0] - (Vdt2[0]*(tau1[0]+tau2[0]))/Vdt12[0])
    c = + (2*Vdt2[0]*tau1[0]*(tau1[0]+tau2[0]))/Vdt12[0]
    d = + Vdt2[0]/Vdt12[0]*(-tau1[0] - (Vdt2[0]*(tau1[0]+tau2[0]))/Vdt12[0])*(tau1[0]+tau2[0])
    e = - Vdt2[0]/Vdt12[0]*(-2*np.power(tau1[0],2)-2*tau1[0]*tau2[0]-2*np.power(tau2[0],2))
    f = + 2*np.power(Vdt2[0]/Vdt12[0],2)*np.power((tau1[0]+tau2[0]),2)
    variance12[0] = np.sum([a,b,c,d,e,f])
    variance12[2] = variance12[0]/np.power(tau12[0],2)              
    
    #%%%% 3rd kumulant = 3nd central moment of the propability distribution = skew12
    '''
    For now these values do not have a propper uncertainty evaluation!
    '''
    skew12[0] = skew12[1].subs({tau1[1]:tau1[0].n, tau2[1]:tau2[0].n, Vdt2[1]:Vdt2[0].n, Vdt12[1]:Vdt12[0].n})
    skew12[0] = float(sym.Float(skew12[0]))
    skew12[0] = ufloat(skew12[0],9999,'For now this value does not have a propper uncertainty evaluation!')
    skew12[2] = skew12[3].subs({tau1[1]:tau1[0].n, tau2[1]:tau2[0].n, Vdt2[1]:Vdt2[0].n, Vdt12[1]:Vdt12[0].n})
    skew12[2] = float(sym.Float(skew12[2]))
    skew12[2] = ufloat(skew12[2],9999,'For now this value does not have a propper uncertainty evaluation!')
    
    #%%%% 4th kumulant lead to 4th central moment of the propability distribution = kurtosis12
    '''
    For now these values do not have a propper uncertainty evaluation!
    '''
    kurtosis12[0] = kurtosis12[1].subs({tau1[1]:tau1[0].n, tau2[1]:tau2[0].n, Vdt2[1]:Vdt2[0].n, Vdt12[1]:Vdt12[0].n})
    kurtosis12[0] = float(sym.Float(kurtosis12[0]))
    kurtosis12[0] = ufloat(kurtosis12[0],9999,'For now this value does not have a propper uncertainty evaluation!')
    kurtosis12[2] = kurtosis12[3].subs({tau1[1]:tau1[0].n, tau2[1]:tau2[0].n, Vdt2[1]:Vdt2[0].n, Vdt12[1]:Vdt12[0].n})
    kurtosis12[2] = float(sym.Float(kurtosis12[2]))
    kurtosis12[2] = ufloat(kurtosis12[2],9999,'For now this value does not have a propper uncertainty evaluation!')
    
    #%% Summarise in a dataframe
    RecycSys = pd.DataFrame([{'tau in h': tau12[0].n, 'std tau in h': tau12[0].s,
                        'sigma² in h²': variance12[0].n, 'std sigma² in h²': variance12[0].s, '(sigma²)* in -': variance12[2].n, 'std (sigma²)* in -': variance12[2].s, 
                        'mu3 in h³': skew12[0].n, 'std mu3 in h³': skew12[0].s, 'mu3* in -': skew12[2].n, 'std mu3* in -': skew12[2].s,
                        'mu4 in h³': kurtosis12[0].n, 'std mu4 in h³': kurtosis12[0].s, 'mu4* in -': kurtosis12[2].n, 'std mu4* in -': kurtosis12[2].s,
                        }])

    return RecycSys

#%% recirculation ratio
def recirculation_ratio(S, S_std,        # volume flow of the ventilated space, in m³/h            
                        epsilonabs, epsilonabs_std,          # volume flow of the short-cut volume, in m³/h
                        ):
    #%% Description
    '''
    See equation 16 of Federspiel '99':
    Federspiel, C. C. (1999), Air-change effectiveness: theory and calculation 
    methods. Indoor Air 9/1, S.47–56, DOI: 10.1111/j.1600-0668.1999.t01-3-00008.x.
    
    Federspiel called its factor "S" short-cut ratio however he redefined it 
    later on in the paper and "modeled it by assuming that ... S<0". In fact
    this is the amount of volume flow stagnating inside the volume.
    
    '''
    
    #%% Neccessary packages
    from uncertainties import ufloat
    
    from sympy import init_printing
    from sympy import Symbol
    
    #%% Define parameters including uncertainty
    S = [ufloat(S, S_std), Symbol(r'S')]
    epsilonabs = [ufloat(epsilonabs, epsilonabs_std), Symbol(r'\varepsilon^{a}')]
    
    #%% Equation
    R = [None]*2
    
    R[0] = (2*epsilonabs[0]+S[0]-1)/(2*epsilonabs[0]*S[0])*100
    R[1] = (2*epsilonabs[1]+S[1]-1)/(2*epsilonabs[1]*S[1])*100
    
    #%% Summarise in a dataframe
    R = pd.DataFrame([{'Rezirkulation in %Vdt': R[0].n, 'std Rezirkulation in %Vdt': R[0].s}])

    return R
# def plot_resitimes(tav):
    
#     pd.options.plotting.backend = "plotly" # NOTE: This changes the plot backend which should be resetted after it is not needed anymore. Otherwise it will permanently cause problems in future, since it is a permanent change.
    
#     import plotly.graph_objects as go
#     fig = go.Figure()
#     fig.add_trace(go.Bar(#title = time["short_name"][t],
#                                   name=r'$\left\langle\bar{t}\right\rangle_\mathrm{3} in \mathrm{3}$',
#                                   x = tav["Experiment"], 
#                                   y = sup_exh_df["av t3 in s"],
#                                   #error_y=dict(value=sup_exh_df["std meas room_av"].max())
#                                   )
#                       )
#     fig.add_trace(go.Scatter(name='calc room_av',
#                                   x = sup_exh_df["datetime"], 
#                                   y = sup_exh_df["calc room_av"],
#                                   #error_y = dict(value=sup_exh_df["std calc room_av"].max())
#                                   )
#                       )
#     fig.add_trace(go.Scatter(name='calc room_exh',
#                                   x = sup_exh_df["datetime"], 
#                                   y = sup_exh_df["calc room_exh"],
#                                   #error_y=dict(value=sup_exh_df["std calc room_exh"].max())
#                                   )
#                       )
#     fig.add_trace(go.Scatter(name='d calc exh-av',
#                                   x = sup_exh_df["datetime"], 
#                                   y = sup_exh_df["d calc exh-av"],
#                                   #error_y=dict(value=sup_exh_df["std d calc exh-av"].max())
#                                   )
#                       )
#     fig.add_trace(go.Scatter(name='supply',x=sup_exh_df["datetime"], y = sup_exh_df["supply"]))
#     fig.add_trace(go.Scatter(name='exhaust',x=sup_exh_df["datetime"], y = sup_exh_df["exhaust"]))
        
#     fig.update_layout(
#             title="{} {}".format(database, aperture_sensor),
#             xaxis_title="Zeit t in hh:mm:ss",
#             yaxis_title=r'Verweilzeit $\bar{t}_1$',
#             legend_title="Legende",
#             font=dict(
#                 family="Segoe UI",
#                 size=18,
#                 color="black"
#             )
#         )
        
#     fig.show()
    
#     import plotly.io as pio
    
#     pio.renderers.default='browser'
#     pd.options.plotting.backend = "matplotlib"
    
#     pass


"""
    Tasks to be done:
        1.) Include uncertainty evaluation for tau_e and tau_s to be returned 
            at the end as well
        2.) Include an option where the plots are turned off by default.
        3.) Only the plots "original [experiment]" and the final plot are
            interesting.

"""


# EE = exchange_efficiency(tau=1.0,  tau_std=99.9,
#                         tav=0.0,  tav_std=9.99,
#                         alphaav=0.0, alphaav_std=9.99,
#                         n=0.0, n_std=9.99,
#                         navex=0.0, navex_std=9.99,
#                         navr=0.0, navr_std=9.99,
#                         sigma_dimless=1.0,  sigma_dimless_std=9.99)

# summary = summary_resitime_vflow(experiment = "W_I_e0_Herdern", reset=False)

# RecycSys = recyclesystem(Vdt12=summary[2]['volume_flow'].loc[0], Vdt12_std=summary[2]['std_volume_flow'].loc[0],     # volume flow entering and leaving the whole system V12, in m³/h           
#                   Vdt2=summary[2]['volume flow Vdt3 in m³/h'].loc[0], Vdt2_std=summary[2]['std volume flow Vdt3 in m³/h'].loc[0],       # volume flow of the backfeed subsystem V2, in m³/h
#                   tau1=summary[10]['tau2 in h'].loc[0], tau1_std=summary[10]['std tau2 in h'].loc[0],   # nominal time constant of subsystem V1, in h
#                   tau2=summary[10]['tau3 in h'].loc[0], tau2_std=summary[10]['std tau3 in h'].loc[0])



# tav2=summary[8]['av t2 in s'].loc[0]
# tav2_std=summary[8]['std av t2 in s'].loc[0]
# Vdt23=summary[2]['volume_flow'].loc[0]
# Vdt23_std=summary[2]['std_volume_flow'].loc[0]
# V23=summary[1]['Volume V23 in m³'].loc[0]
# V23_std=summary[1]['std Volume V23 in m³'].loc[0]
# alphaav3=summary[3]['av restime_3 in h'].loc[0]/2
# alphaav3_std=summary[3]['std av restime_3 in h'].loc[0]/2


# V2 = short_cut_volume(tav2 = tav2, tav2_std = tav2_std,            
#                       Vdt23 = Vdt23, Vdt23_std = Vdt23_std,          
#                       V23 = V23, V23_std = V23_std,             
#                       alphaav3 = alphaav3, alphaav3_std = alphaav3_std)

experiments = ["S_I_e0_Herdern", "S_H_e0_Herdern", "W_I_e0_Herdern","W_H_e0_Herdern",
                "S_H_e0_ESHL","S_I_e0_ESHL",] #"W_H_e0_ESHL", "W_I_e0_ESHL"

summaryE = []
reistimesE = pd.DataFrame(columns=([0,
  'av restime_3 in h',
  'std av restime_3 in h',
  'av t1 in s',
  'std av t1 in s',
  'av t2 in s',
  'std av t2 in s']))
ratiosE = pd.DataFrame(columns=([0,
                                  'Kurzschluss in %Vdt', 'std Kurzschluss in %Vdt',
                                  'Stagnation in %Vdt', 'std Stagnation in %Vdt']))
EEVdt = pd.DataFrame(columns=([0,
                                  'epsilon in %t', 'std epsilon in %t',
                                  'volume_flow', 'std_volume_flow',
                                  'n in 1/h', 'std n in 1/h']))

for e in experiments:
    summary = summary_resitime_vflow(e)
    new_row = pd.DataFrame(columns=([0,
          'av restime_3 in h',
          'std av restime_3 in h',
          'av t1 in s',
          'std av t1 in s',
          'av t2 in s',
          'std av t2 in s']))
    new_row = pd.concat([pd.Series(summary[0]), summary[3], summary[7], summary[8]], join="outer", axis=1)
    reistimesE = reistimesE.append(new_row)
    
    new_row2 = pd.DataFrame(columns=([0,
                                  'Kurzschluss in %Vdt', 'std Kurzschluss in %Vdt',
                                  'Stagnation in %Vdt', 'std Stagnation in %Vdt']))
    new_row2 = pd.concat([pd.Series(summary[0]), summary[9]], join="outer", axis=1)
    ratiosE = ratiosE.append(new_row2)
    new_row3 = pd.DataFrame(columns=([0,
                                  'epsilon in %t', 'std epsilon in %t',
                                  'volume_flow', 'std_volume_flow',
                                  'n in 1/h', 'std n in 1/h']))
    new_row3 = pd.concat([pd.Series(summary[0]), summary[2].iloc[:,[0,1]], summary[12].iloc[:,[0,1,8,9]]], join="outer", axis=1)
    EEVdt = EEVdt.append(new_row3)
    
    summaryE.append(summary)

# reistimesE['av restime_3 in h'] = reistimesE['av restime_3 in h'] * 3600
# reistimesE['std av restime_3 in h'] = reistimesE['std av restime_3 in h'] * 3600
# reistimesE = reistimesE.rename(columns={0: 'Experiment','av restime_3 in h': 'av t3 in s', 'std av restime_3 in h': 'std av t3 in s'})
# reistimesE = reistimesE.reset_index()
# del reistimesE['index']

# ratiosE = ratiosE.reset_index()
# del ratiosE['index']
    


# # summary = summary_resitime_vflow(experiment = "W_I_e0_Herdern")

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

