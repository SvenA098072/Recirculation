# -*- coding: utf-8 -*-
"""
Created on Thu Apr  8 14:26:27 2021

@author: Devineni and Sven finally merged
"""
# Necessary modules
import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
pd.options.plotting.backend = "matplotlib" # NOTE: This is useful in case the plotbackend has been changed by any previously (even befor machine shut-downs).

# from statistics import mean
from tabulate import tabulate
from sqlalchemy import create_engine


# functions to print in colour
def prRed(skk): print("\033[31;1;m {}\033[00m" .format(skk)) 
def prYellow(skk): print("\033[33;1;m {}\033[00m" .format(skk)) 
# The following is a general syntax to dedine a MySQL connection
# =============================================================================
# engine = create_engine("mysql+pymysql://admin:the_secure_password_4_ever@localhost/",\
#                          pool_pre_ping=True) # Krishna's local address
# =============================================================================
engine = create_engine("mysql+pymysql://wojtek:Password#102@wojtek.mysql.database.azure.com/",\
                      pool_pre_ping=True) # Cloud server address

def residence_time_sup_exh(experimentno=16, deviceno=1, periodtime=120, plot=False):
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
    
    #%% Load relevant data
    t = experimentno # to select the experiment (see Timeframes.xlsx)
    l = deviceno # to select the sensor in the ventilation device
    
    T = periodtime                                                              # T in s; period time of the ventilation systems push-pull devices.
    # time = pd.read_excel("C:/Users/Devineni/OneDrive - bwedu/4_Recirculation/Times_thesis.xlsx", sheet_name="Timeframes")
    # The dataframe time comes from the excel sheet in the path above, to make -
    # - changes go to this excel sheet, edit and upload it to mysql.
    
    time = pd.read_sql_query("SELECT * FROM testdb.timeframes;", con = engine)      #standard syntax to fetch a table from Mysql
    
    start, end = str(time["Start"][t] - dt.timedelta(minutes=20)), str(time["End"][t]) # selects start and end times to slice dataframe
    t0 = time["Start"][t]                                                           #actual start of the experiment
    
    table = time["tables"][t].split(",")[l]                                         #Name of the ventilation device
    
    dum = [["Experiment",time["short_name"][t] ], ["Sensor", table]]                # Creates a list of 2 rows filled with string tuples specifying the experiment and the sensor.
    print(tabulate(dum))                                                            # Prints the inut details in a table
    
    database = time["database"][t]                                                  # Selects the name of the database as a string 
    
    
    background, dummy = outdoor(str(t0), str(end), plot = False)                    # Syntax to call the background concentration function, "dummy" is only necessary since the function "outdoor" returns a tuple of a dataframe and a string.
    background = background["CO2_ppm"].mean()                                       # Future: implement cyclewise background concentration; Till now it takes the mean outdoor concentration of the whole experiment.
    
    df = pd.read_sql_query("SELECT * FROM {}.{} WHERE datetime BETWEEN '{}' AND\
                           '{}'".format(database, table, start, end), con = engine)
    df = df.loc[:,["datetime", "CO2_ppm"]]
    df["original"] = df["CO2_ppm"]                                                  # filters only the CO2 data till this line
    df.columns = ["datetime", "original", "CO2_ppm"]
    
    df["original"] = df["CO2_ppm"]                                                  # Copies the original absolute CO2-concentrations data form CO2_ppm in a "backup"-column originals 
    
    df["CO2_ppm"] = df["CO2_ppm"] - background                                      # substracts the background concentrations -> CO2_ppm contains CO2-concentration of some instance of time above background concentration.
    
    if df["CO2_ppm"].min() < 0:                                                     # Sometimes the accumulated amount of CO2 concentraion becomes negative. This is not possible and would lead to a mistake for the integral calculation. An artificial offset lifts the whole decay curve at >=0.
        offset = df["CO2_ppm"].min()
        df["CO2_ppm"] = df["CO2_ppm"] - offset
        
    df = df.loc[~df.duplicated(subset=["datetime"])]                                # Checks for duplicated in datetime and removed them; @Krishna: How can such a duplicate occur?
    diff = (df["datetime"][1]-df["datetime"][0]).seconds                            # integer diff in s; Calculates the length of the time interval between two timestamps 
    df = df.set_index("datetime")                                                   # Resets the index of the dataframe df from the standard integer {0, 1, 2, ...} to be exchanged by the datetime column containing the timestamps.
    
    while not(t0 in df.index.to_list()):                                            # The t0 from the excel sheet may not be precice that the sensor starts 
        t0 = t0 + dt.timedelta(seconds=1)                                           # - starts at the same time so i used this while loop to calculate the 
        print(t0)                                                                   # - the closest t0 after the original t0
    
    df["roll"] = df["CO2_ppm"].rolling(int(T/diff)).mean()                          # moving average for 2 minutes, used to calculate Cend; T = 120s is the period time of the push-pull ventilation devices which compose the ventilation system. 
    
    
    
    c0 = df["CO2_ppm"].loc[t0]                                                      # C0; @DRK: Check if c0 = df["roll"].loc[t0] is better here.
    Cend37 = round((c0)*0.37, 2)                                                    # @DRK: From this line 101 schould be changed.   
    
    cend = df.loc[df["roll"].le(Cend37)]                                            # Cend: Sliced df of the part of the decay curve below the 37 percent limit
    
    if len(cend) == 0:                                                              # Syntax to find the tn of the experiment
        tn = str(df.index[-1])
        print("The device has not reached 37% of its initial concentration")
    else:
        tn = str(cend.index[0])
    
   
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
        df.plot(title = "original " + time["short_name"][t], color = [ 'silver', 'green', 'orange'], ax = ax)
        df['max'].plot(marker='o', ax = ax)                                             # This needs to be verified with the graph if python recognizes all peaks
        df['min'].plot(marker="v", ax = ax)                                             # - and valleys. If not adjust the n value.
    else:
        pass
    
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
    
    #%%% Plot Matplotlib                                                            # This can be verified from this graph        
# =============================================================================
#     if plot:
#         #%%%% supply
#         plt.figure()
#         a["CO2_ppm"].plot(title = "supply " + time["short_name"][t]) 
#         a["CO2_ppm"].plot(title = "supply") 
#           
#         #%%%% exhaust
#         b["CO2_ppm"].plot(title = "exhaust " + time["short_name"][t])                                            # Similar procedure is repeated from exhaust
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
    
        sup_exh_df = pd.concat([df_sup2, df_exh2], axis = 1).reset_index()
        sup_exh_df.columns = ["datetime","supply", "exhaust"]
        
        import plotly.express as px
        fig = px.line(sup_exh_df, x="datetime", y = sup_exh_df.columns, title = time["short_name"][t])
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
    
    start_date = str(t0); end_date = tn # CHANGE HERE 
    
    mask = (df_sup3['datetime'] > start_date) & (df_sup3['datetime'] <= end_date)
    
    df_sup3 = df_sup3.loc[mask]
    
    
    for i,j in df_sup3.iterrows():                                                  # *.interrows() will always return a tuple encapsulating an int for the index of the dataframe where it is applied to and a series containing the data of row selected. Therefore it is good to seperate both before in e.g. i,j .
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
        
    
    df_sup_list = []
    for i in range(1, int(df_sup3.num.max()+1)):
        df_sup_list.append(df_sup3.loc[df_sup3["num"]==i])
    
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

    res = reg_result[reg_result['sensor'].str.lower() == table].reset_index(drop = True)
    accuracy1 = 50 # it comes from the equation of uncertainity for testo 450 XL
    accuracy2 = 0.02 # ±(50 ppm CO2 ±2% of mv)(0 to 5000 ppm CO2 )
            
    accuracy3 = 50 # the same equation for second testo 450 XL
    accuracy4 = 0.02
            
    accuracy5 = 75 # # the same equation for second tes
    accuracy6 = 0.03 # Citavi Title: Testo AG
    
    df_tau_sup = []
    for idf in df_sup_list:
        if len(idf) > 3:
            a = idf.reset_index(drop = True)
            a['CO2_ppm_reg'] = a.eval(res.loc[0, "equation"])    
            a = a.rename(columns = {'CO2_ppm':'CO2_ppm_original', 'CO2_ppm_reg': 'CO2_ppm'})
            a = a.drop_duplicates(subset=['datetime'])
            a = a.loc[:, ["datetime", "CO2_ppm_original", "CO2_ppm"]]
            a = a.dropna()
            a["log"] = np.log(a["CO2_ppm"])
            
            diff = (a["datetime"][1] - a["datetime"][0]).seconds
            a["s_meas"] =  np.sqrt(np.square((a["CO2_ppm"] * accuracy2)) 
                                   + np.square(accuracy1) + np.square((a["CO2_ppm"] * accuracy4)) 
                                   + np.square(accuracy3) + np.square((a["CO2_ppm"] * accuracy6)) 
                                   + np.square(accuracy5)+ np.square(res.loc[0, "rse"]))
            ns_meas = a['s_meas'].mean()
            n = len(a['s_meas'])
            global sa_num, s_lambda, s_phi_e
            
            ### ISO 16000-8 option to calculate slope (defined to be calculated by Spread-Sheat/Excel)
            a["runtime"] = np.arange(0,len(a) * diff, diff)
            
            a["t-te"] = a["runtime"] - a["runtime"][len(a)-1]
            
            a["lnte/t"] = a["log"] - a["log"][len(a)-1]
            
            a["slope"] = a["lnte/t"] / a["t-te"]
            slope_sup = a["slope"].mean()
            
            ### More acurate option to calculate the solpe of each (sub-)curve
            x1 = a["runtime"].values
            y1 = a["log"].values
            from scipy.stats import linregress
            slope = linregress(x1,y1)[0]
            ###
            
            a.loc[[len(a)-1], "slope"] = abs(slope_sup)
            
            sumconz = a["CO2_ppm"].iloc[1:-1].sum()
            
            tail = a["CO2_ppm"][len(a)-1]/abs(slope_sup)
            area_sup_1= (diff * (a["CO2_ppm"][0]/2 + sumconz +a["CO2_ppm"][len(a)-1]/2))
            from numpy import trapz
            global area_sup_2, s_rest, s_total, a_rest, a_tot,sa_num,s_lambda, s_phi_e,s_rest, sa_rest, s_area
            
            area_sup_2 = trapz(a["CO2_ppm"].values, dx=diff) # proof that both methods have same answer
            
            a_rest = a["CO2_ppm"].iloc[-1]/abs(slope)
            a_tot = area_sup_2 + a_rest
            
            sa_num = ns_meas * (diff) * ((n - 1)/np.sqrt(n)) # Taken from DIN ISO 16000-8:2008-12, Equation D2 units are cm3.m-3.sec
            s_lambda = a["slope"][:-1].std()/abs(a["slope"][:-1].mean())
            s_phi_e = a["slope"][:-1].std()/abs(a["slope"].iloc[-1])
    
            s_rest = np.sqrt(pow(s_lambda,2) + pow(s_phi_e,2))
            sa_rest = s_rest * a_rest
            s_area = np.sqrt(pow(sa_num,2) + pow(sa_rest,2))/a_tot
            s_total = np.sqrt(pow(s_area,2) + pow(0.05,2))
    
    
            tau = (area_sup_2 + tail)/a["CO2_ppm"][0]
            a["tau_sec"] = tau
            a.loc[:, "s_total"] = s_total

            df_tau_sup.append(a)
        else:
            pass
    #%%% Supply final tau
    tau_list_sup = []
    for h in df_tau_sup:
        tau_list_sup.append(h["tau_sec"][0])
    
    tau_s = np.mean(tau_list_sup)
    
        
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
    
    df_exh_list = []
    for i in range(1, int(df_exh3.num.max()+1)):
        df_exh_list.append(df_exh3.loc[df_exh3["num"]==i])
    #%%% Exhaust tau
    # this method can be replicated in Excel for crossverification
    """Calculates tau based in area under the curve"""
    
    df_tau_exh = []
    for e in df_exh_list:
        if len(e) > 3:
            b = e.reset_index(drop = True)
            
            
            b['CO2_ppm_reg'] = b.eval(res.loc[0, "equation"])    
            b = b.rename(columns = {'CO2_ppm':'CO2_ppm_original', 'CO2_ppm_reg': 'CO2_ppm'})
            b = b.drop_duplicates(subset=['datetime'])
            b = b.loc[:, ["datetime", "CO2_ppm_original", "CO2_ppm"]]
            b = b.dropna()
            
            
            
            b["log"] = np.log(b["CO2_ppm"])
            
            diff = (b["datetime"][1] - b["datetime"][0]).seconds
            b["s_meas"] =  np.sqrt(np.square((b["CO2_ppm"] * accuracy2)) 
                                   + np.square(accuracy1) + np.square((b["CO2_ppm"] * accuracy4)) 
                                   + np.square(accuracy3) + np.square((b["CO2_ppm"] * accuracy6)) 
                                   + np.square(accuracy5)+ np.square(res.loc[0, "rse"]))
            ns_meas = b['s_meas'].mean()
            n = len(b['s_meas'])
            
            
            b["runtime"] = np.arange(0,len(b) * diff, diff)
            
            b["t-te"] = b["runtime"] - b["runtime"][len(b)-1]
            
            b["lnte/t"] = b["log"] - b["log"][len(b)-1]
            
            b["slope"] = b["lnte/t"] / b["t-te"]
            slope = b["slope"].mean()
            #############
            x = b["runtime"].values
            y = b["log"].values
            from scipy.stats import linregress
            slope = linregress(x,y)[0]
            ############
            b.loc[[len(b)-1], "slope"] = abs(slope)
            
            sumconz = b["CO2_ppm"].iloc[1:-1].sum()
            
            tail = b["CO2_ppm"][len(b)-1]/abs(slope)
            area1 = (diff * (b["CO2_ppm"][0]/2 + sumconz +b["CO2_ppm"][len(b)-1]/2))
            from numpy import trapz
            global area2
            area2 = trapz(b["CO2_ppm"].values, dx=diff)                             # proof that both methods have same answer
            
           
            
            a_rest = b["CO2_ppm"].iloc[-1]/abs(slope)
            a_tot = area2 + a_rest
            
            sa_num = ns_meas * (diff) * ((n - 1)/np.sqrt(n)) # Taken from DIN ISO 16000-8:2008-12, Equation D2 units are cm3.m-3.sec
            s_lambda = b["slope"][:-1].std()/abs(b["slope"][:-1].mean())
            s_phi_e = b["slope"][:-1].std()/abs(b["slope"].iloc[-1])
    
            s_rest = np.sqrt(pow(s_lambda,2) + pow(s_phi_e,2))
            sa_rest = s_rest * a_rest
            s_area = np.sqrt(pow(sa_num,2) + pow(sa_rest,2))/a_tot
            s_total = np.sqrt(pow(s_area,2) + pow(0.05,2))
            
            
            tau2 = (area2 + tail)/b["CO2_ppm"][len(b)-1]
            b["tau_sec"] = tau2
            b.loc[:, "s_total"] = s_total
            df_tau_exh.append(b)
        else:
            pass
    #%%% Exhaust final tau 
    tau_list_exh = []
    for jdf in df_tau_exh:
        tau_list_exh.append(jdf["tau_sec"][0])
    
    tau_e = np.mean(tau_list_exh)
    
    #%% returned values
    """
        Returns:
            t0 = initial timestamp of the start of the experiment
            tn = final timestamp of the evaluated data
            tau_e = exhaust residence time of the short-cut volume
            tau_s = exhaust residence time of the recirculation volume 
                    ("supply residence time")
    """
    tn = cend.index[0]
    return [t0, tn, tau_e, tau_s]

    #%% Final result print
    prYellow("Recirculation:  {} %".format(round((tau_s/tau_e)*100) )  )


"""
    Tasks to be done:
        1.) Include uncertainty evaluation for tau_e and tau_s to be returned 
            at the end as well
        2.) Include an option where the plots are turned off by default.
        3.) Only the plots "original [experiment]" and the final plot are
            interesting.

"""

residence_time_sup_exh()