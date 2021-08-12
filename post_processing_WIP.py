# -*- coding: utf-8 -*-
"""
Created on Thu Jun 24 13:28:14 2021

@author: Raghavakrishna

Copy of the "post_processing.py" from 08th Aug 2021
"""


import pandas as pd
from datetime import timedelta
import datetime as dt
import statistics
import datetime

from sqlalchemy import create_engine
import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.units as munits

from uncertainties import ufloat
from uncertainties import unumpy
from uncertainties import umath

import tabulate

def prRed(skk): print("\033[31;1;m {}\033[00m" .format(skk))
def prYellow(skk): print("\033[33;1;m {}\033[00m" .format(skk))


#%%
class ResidenceTimeMethodError(ValueError):
    def __str__(self):
        return 'You need to select a valid method: iso, trapez or simpson (default)'

#%%
class CBO_ESHL:
  
    def __init__(self, experiment = "W_I_e0_Herdern", 
                 testo_sensor = "2a_testo",
                 aperture_sensor = "2l",
                 column_name = 'hw_m/sec'):
        
        """
        Takes 3 inputs some of them are not necessary for certain methods.

        Input Parameters
        ----------
            experiment : str 
                The name of the experiment available in the master_time_sheet.xlsx or in my thesis.
            testo_sensor : str
                the type of nomenclature used to describe Testo sensors. This 
                input is required to evaluate the Testo data like wind speed
            column_name : str
                The column name used when saving testo data. the column name 
                is also an indication of the units used for the measured parameter.
                This alrady describes what sensor is measuring.
                
        Imported Parameters
        ----------
            t0 : datetime 
                The actual start of the experiment.
            tn : datetime
                The approximate end of the experiemnt.
            tau_nom : float
                The nominal time constant of the measurement obtained from 
                master_time_sheet.xlsx
        """
        excel_sheet = "master_time_sheet.xlsx"
        self.times = pd.read_excel(excel_sheet, sheet_name = "Sheet1")
        self.input = pd.read_excel(excel_sheet, sheet_name = "inputs")
        self.experiment = experiment
        self.testo_sensor = testo_sensor
        self.column_name = column_name
        #self.engine = create_engine("mysql+pymysql://wojtek:Password#102@wojtek.mysql.database.azure.com/",pool_pre_ping=True)
        self.engine = create_engine("mysql+pymysql://root:Password123@localhost/",pool_pre_ping=True)
        self.aperture_sensor = aperture_sensor
        self.database = self.times[self.times["experiment"] == self.experiment].iloc[0,3]
        self.t0 = self.times[self.times["experiment"] == experiment].iloc[0,1]
        self.tn = self.times[self.times["experiment"] == experiment].iloc[0,2]
        self.exclude = self.times[self.times["experiment"] == experiment].iloc[0,4].split(",")
        
        self.calibration = self.times[self.times["experiment"] == experiment].iloc[0,5]
        # self.engine1 = create_engine("mysql+pymysql://wojtek:Password#102@wojtek.mysql.database.azure.com/{}".format(self.calibration),pool_pre_ping=True)
        self.engine1 = create_engine("mysql+pymysql://root:Password123@localhost/{}".format(self.calibration),pool_pre_ping=True)

        self.wall_database = self.times[self.times["experiment"] == experiment].iloc[0,6]
        self.testos = ["1a_testo","2a_testo","3a_testo","4a_testo"]
        
        self.t0_20 = self.t0 - timedelta(minutes = 20)
        self.tn_20 = self.tn + timedelta(minutes = 20)   

        self.tau_nom = self.input.loc[self.input["experiment"] == self.experiment]["tau_nom"].iat[0]                             
        
    def wind_velocity_indoor(self):
        """
        Prints the person's name and age.

        If the argument 'additional' is passed, then it is appended after the main info.

        Parameters
        ----------
        additional : str, optional
            More info to be displayed (default is None)

        Returns
        -------
        None
        """
        self.df1 = pd.read_sql_query("SELECT * FROM {}.{} WHERE datetime BETWEEN '{}' AND '{}'".format(self.database, self.testo_sensor, self.t0_20, self.tn_20), con = self.engine) 
        self.df2 = self.df1.loc[:, ["datetime", "hw_m/sec"]]
        self.df2 = self.df2.set_index("datetime")
        self.df2 = self.df2.truncate(str(self.t0), str(self.tn) )
        
        self.stats = self.df2.describe().iloc[[0,1,2,3,7],:]
        self.stats.columns = ["values"]
        
        self.data = {"values":[self.experiment, self.sensor_name, "hw_m/sec", self.t0, self.tn]}
        
        self.empty_df = pd.DataFrame(self.data, index =['experiment',
                                        'sensor name',
                                        'column name', "Start", "End"])
         
        self.res = pd.concat([self.empty_df, self.stats], axis = 0)
        
        return self.res
    
    
    
    def wind_velocity_outdoor(self):
        
        self.df1 = pd.read_sql_query("SELECT * FROM weather.weather_all WHERE datetime BETWEEN '{}' AND '{}'".format( self.t0_20, self.tn_20), con = self.engine) 
        self.df2 = self.df1.loc[:, ["datetime", "Wind Speed, m/s", "Gust Speed, m/s", "Wind Direction"]]
        self.df2 = self.df2.set_index("datetime")
        self.df2 = self.df2.truncate(str(self.t0), str(self.tn) )
        
        self.stats = self.df2.describe().iloc[[0,1,2,3,7],:]
        self.empty_df = pd.DataFrame(index =['experiment',
                                  'table name', 'Start', 'End'],
                    columns =["Wind Speed, m/s", "Gust Speed, m/s", "Wind Direction"])
        
        self.empty_df.loc["experiment", ["Wind Speed, m/s","Gust Speed, m/s", "Wind Direction"]] = self.experiment
        self.empty_df.loc["table name", ["Wind Speed, m/s","Gust Speed, m/s", "Wind Direction"]] = "weather_all"
        self.empty_df.loc["Start", ["Wind Speed, m/s","Gust Speed, m/s", "Wind Direction"]] = self.t0
        self.empty_df.loc["End", ["Wind Speed, m/s","Gust Speed, m/s", "Wind Direction"]] = self.tn

        self.res = pd.concat([self.empty_df, self.stats], axis = 0)
        
      
        return self.res
    

    def aussen(self, plot = False, save = False):
        """
        This method calculates the outdoor CO2 concentration from the HOBO sensor
        ALso this produces a graph of outdoor CO2 data which is rolled for 120 seconds

        Parameters
        ----------
        plot : BOOL, optional
            if True displays a graph. The default is False.
        save : BOOL, optional
            If True saves in the current directory. The default is False. 
            You can also change the plot saving and rendering settings in the code

        Returns
        -------
        dictionary
            The dictionary contains the mean , std , max and min of CO2 for the 
            experimental period.

        """
        if self.experiment == "S_I_e0_Herdern" or self.experiment == "S_I_e1_Herdern":
            self.Cout = {'meanCO2': 445.1524174626867,
                    'sgm_CO2': 113.06109664245112,
                    'maxCO2': 514.3716999999999,
                    'minCO2': 373.21639999999996}
            self.cout_mean, self.cout_max, self.cout_min = 445.1524174626867, 514.3716999999999, 373.21639999999996
            
            if plot:
                print("The outdoor plot for this experiment is missing due to lack of data")
            
            return self.Cout
        else:
            
            accuracy1 = 50 # it comes from the equation of uncertainity for testo 450 XL
            accuracy2 = 0.02 # ±(50 ppm CO2 ±2% of mv)(0 to 5000 ppm CO2 )
            
            accuracy3 = 50 # the same equation for second testo 450 XL
            accuracy4 = 0.02
            
            accuracy5 = 75 # # the same equation for second testo 480
            accuracy6 = 0.03 # Citavi Title: Testo AG
            
            '''
            The following if esle statement is writtten to import the right data 
            for calibration offset equation
            There are two time periods where calibration was done and this
            '''

            
            '''standard syntax to import sql data as dataframe
            self.engine is measurement campagin experimentl data and engine1 is calibration data'''

            '''Calibration data is imported '''
            reg_result = pd.read_sql_table("reg_result", con = self.engine1).drop("index", axis = 1)
            '''Calibration data for the particular sensor alone is filtered '''
            res = reg_result[reg_result['sensor'].str.lower() == "außen"].reset_index(drop = True)
            
            '''This is to filter the HOBOs from testos, The hobos will have a res variable Testos will not have
            because they dont have experimantal calibration offset'''
            if res.shape[0] == 1:
                ''' The imported sql data is cleaned and columns are renamed to suit to out calculation'''
                self.sensor_df3 = pd.read_sql_query("SELECT * FROM {}.{} WHERE datetime BETWEEN '{}' AND '{}'".format(self.database, "außen", self.t0, self.tn) , self.engine).drop('index', axis =1)
                self.sensor_df3['CO2_ppm_reg'] = self.sensor_df3.eval(res.loc[0, "equation"])    
                self.sensor_df3_plot = self.sensor_df3.copy()
                
                
                self.sensor_df3 = self.sensor_df3.rename(columns = {'CO2_ppm':'CO2_ppm_original', 'CO2_ppm_reg': 'C_CO2 in ppm'})
                self.sensor_df3 = self.sensor_df3.drop_duplicates(subset=['datetime'])
                self.sensor_df3 = self.sensor_df3.loc[:, ["datetime", "C_CO2 in ppm", "CO2_ppm_original"]]
                self.sensor_df3 = self.sensor_df3.dropna()
                '''This is the absolute uncertainity at each point of measurement saved in the
                dataframe at each timestamp Ref: equation D2 in DIN ISO 16000-8:2008-12'''
                
                
                '''For ESHL summer ideally we take mean of all three sensors and also propogate 
                the uncertainities of al three testo sensors, This is not done here at the moment
                But to get the most uncertainity possible we peopogte the uncertainity first'''
                # Why RSE ? https://stats.stackexchange.com/questions/204238/why-divide-rss-by-n-2-to-get-rse
                self.sensor_df3["s_meas"] =  np.sqrt(np.square((self.sensor_df3["C_CO2 in ppm"] * accuracy2)) + np.square(accuracy1) + np.square((self.sensor_df3["C_CO2 in ppm"] * accuracy4)) + np.square(accuracy3) + np.square((self.sensor_df3["C_CO2 in ppm"] * accuracy6)) + np.square(accuracy5)+ np.square(res.loc[0, "rse"])) 
                # Die Messunsicherheit hängt sicher in einem bestimmten Umfang vom Konzentrationsbereich ab.DIN ISO 16000-8:2008-12 (page 36)
        
                x = self.sensor_df3["datetime"][2] - self.sensor_df3["datetime"][1]
                self.sec3 = int(x.total_seconds())
                
                if plot:
                    
                    self.sensor_df3_plot = self.sensor_df3_plot.loc[:,['datetime', 'temp_°C', 'RH_%rH', 'CO2_ppm_reg']]
                    self.sensor_df3_plot = self.sensor_df3_plot.set_index("datetime")
                    self.sensor_df3_plot = self.sensor_df3_plot.rolling(int(120/self.sec3)).mean()
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
                    par2.spines["right"].set_position(("axes", 1.2))
                    # Having been created by twinx, par2 has its frame off, so the line of its
                    # detached spine is invisible.  First, activate the frame but make the patch
                    # and spines invisible.
                    make_patch_spines_invisible(par2)
                    # Second, show the right spine.
                    par2.spines["right"].set_visible(True)
                    
                    p1, = host.plot(self.sensor_df3_plot.index, self.sensor_df3_plot['temp_°C'], "b-", label="Temperature (°C)", linewidth=1)
                    p2, = par1.plot(self.sensor_df3_plot.index, self.sensor_df3_plot['CO2_ppm_reg'], "r--", label="CO2 (ppm)", linewidth=1)
                    p3, = par2.plot(self.sensor_df3_plot.index, self.sensor_df3_plot['RH_%rH'], "g-.", label="RH (%)", linewidth=1)
                    
                    # host.set_xlim(0, 2)
                    host.set_ylim(0, 30)
                    par1.set_ylim(0, 3000)
                    par2.set_ylim(0, 100)
                    
                    host.set_xlabel("Time")
                    host.set_ylabel("Temperature (°C)")
                    par1.set_ylabel(r'$\mathrm{CO_2 (ppm)} $')
                    par2.set_ylabel("RH (%)")
                    
                    host.yaxis.label.set_color(p1.get_color())
                    par1.yaxis.label.set_color(p2.get_color())
                    par2.yaxis.label.set_color(p3.get_color())
                    
                    tkw = dict(size=4, width=1.5)
                    host.tick_params(axis='y', colors=p1.get_color(), **tkw)
                    par1.tick_params(axis='y', colors=p2.get_color(), **tkw)
                    par2.tick_params(axis='y', colors=p3.get_color(), **tkw)
                    host.tick_params(axis='x', **tkw)
                
                    import matplotlib.dates as mdates
                    locator = mdates.AutoDateLocator(minticks=3, maxticks=11)
                    formatter = mdates.ConciseDateFormatter(locator)
                    host.xaxis.set_major_locator(locator)
                    host.xaxis.set_major_formatter(formatter)
                    
                    lines = [p1, p2, p3]
                    
                    plt.title("Outdoor data for {}".format(self.experiment))
                    
                    host.legend(lines, [l.get_label() for l in lines])
                    if save:
                        plt.savefig('{} outdoor data (HOBO)'.format(self.experiment), bbox_inches='tight', dpi=400)
                
                    plt.show()
                """
                Creating a runtime column with t0 as 0 or centre of the time axes
                """
                t0_cd = self.sensor_df3['datetime'].loc[0]
                
                while not(self.t0 in self.sensor_df3["datetime"].to_list()):
                    self.t0 = self.t0 + dt.timedelta(seconds=1)
                    # print(self.t0)
                    
                dtl_t0 = (self.t0 - t0_cd)//dt.timedelta(seconds=1)
                
                """
                Calucates the elapsed time stored in the array x as an interger of seconds
                """
                endpoint = len(self.sensor_df3) * self.sec3 - dtl_t0
                
                """
                Creates an array starting with 0 till endpoint with stepsize sec3.
                """
                x = np.arange(-dtl_t0,endpoint,self.sec3)
                
                self.sensor_df3['runtime'] = x
                
                self.sensor_df2 = self.sensor_df3.set_index('datetime')
                self.rhg = pd.date_range(self.sensor_df2.index[0], self.sensor_df2.index[-1], freq=str(self.sec3)+'S')   
                self.au_mean = self.sensor_df2.reindex(self.rhg).interpolate()
                
                self.au_mean['C_CO2 in ppm_out'] = self.au_mean['C_CO2 in ppm']
                self.cout_max = self.au_mean['C_CO2 in ppm_out'].max()
                self.cout_min = self.au_mean['C_CO2 in ppm_out'].min()
                self.cout_mean = self.au_mean['C_CO2 in ppm_out'].mean()
                
                """
                The default value (499±97)ppm (kp=2) has been calculated as the average CO2-
                concentration of the available outdoor measurement data in 
                ...\CO2-concentration_outdoor\.
                However the value should be setted as a list of datapoints for the natural
                outdoor concentration for a time inverval covering the measurement interval.
                
                In future it would be great to have a dataframe with CO2-concentrations for 
                coresponding time stamps.
                """
                self.Cout = {'meanCO2':self.cout_mean, 
                        'sgm_CO2':self.au_mean["s_meas"].mean(), # More clarification needed on uncertainity
                        'maxCO2':self.cout_max,
                        'minCO2':self.cout_min}
                return self.Cout
        
    
    
    def mean_curve(self, plot = False, method='simpson'):
        """
        method:
            'iso'       (Default) The method described in ISO 16000-8 will be applied
                        however this method has a weak uncertainty analysis.
            'trapez'    corrected ISO 16000-8 method applying the trapezoidal method
                        for the interval integration and considers this in the 
                        uncertainty evaluation.
            'simpson'   Applies the Simpson-Rule for the integration and consequently 
                        considers this in the uncertainty evaluation.
        """
        self.names = pd.read_sql_query('SHOW TABLES FROM {}'.format(self.database), con = self.engine)
        self.names = self.names.iloc[:,0].to_list()
        self.new_names = [x for x in self.names if (x not in self.exclude)]
        
        accuracy1 = 50 # it comes from the equation of uncertainity for testo 450 XL
        accuracy2 = 0.02 # ±(50 ppm CO2 ±2% of mv)(0 to 5000 ppm CO2 )
        
        accuracy3 = 50 # the same equation for second testo 450 XL
        accuracy4 = 0.02
        
        accuracy5 = 75 # # the same equation for second testo 480
        accuracy6 = 0.03 # Citavi Title: Testo AG
        
        
        self.cdf_list, self.df_tau, self.tau_hr, self.s_total_abs_hr  = [], [], [], []
        for self.table in self.new_names:
            self.cdf1 = pd.read_sql_query("SELECT * FROM {}.{} WHERE datetime BETWEEN '{}' AND '{}'".format(self.database, self.table, self.t0, self.tn), con = self.engine) 
            self.cdf2 = self.cdf1.loc[:,["datetime", "CO2_ppm"]]
            self.reg_result = pd.read_sql_table("reg_result", con = self.engine1).drop("index", axis = 1)
            '''Calibration data for the particular sensor alone is filtered '''
            self.res = self.reg_result[self.reg_result['sensor'].str.lower() == self.table].reset_index(drop = True)
            
            if "testo" not in self.table:
                self.cdf2['CO2_ppm_reg'] = self.cdf2.eval(self.res.loc[0, "equation"]) 
                self.cdf2 = self.cdf2.rename(columns = {'CO2_ppm':'CO2_ppm_original', 'CO2_ppm_reg': 'CO2_ppm'})
                self.cdf2 = self.cdf2.drop_duplicates(subset=['datetime'])
                self.cdf2 = self.cdf2.loc[:, ["datetime", "CO2_ppm"]]
                self.cdf2 = self.cdf2.dropna()
            
            if self.cdf2["CO2_ppm"].min() < self.aussen()["meanCO2"]:
                self.cdf2.loc[:,"CO2_ppm"] = self.cdf2.loc[:,"CO2_ppm"] - (self.cdf2.loc[:,"CO2_ppm"].min() - 3)
            else:
                self.cdf2.loc[:,"CO2_ppm"] = self.cdf2.loc[:,"CO2_ppm"] - self.aussen()["meanCO2"]
            
            self.cdf2 = self.cdf2.fillna(method="bfill", limit=2)
            self.cdf2 = self.cdf2.fillna(method="pad", limit=2)
            
            self.cdf2.columns = ["datetime", str(self.table)]
            self.cdf2["log"] = np.log(self.cdf2[str(self.table)])
            self.diff_sec = (self.cdf2["datetime"][1] - self.cdf2["datetime"][0]).seconds
            self.cdf2["s_meas"] =  np.sqrt(np.square((self.cdf2[str(self.table)] * accuracy2)) 
                                   + np.square(accuracy1) + np.square((self.cdf2[str(self.table)] * accuracy4)) 
                                   + np.square(accuracy3) + np.square((self.cdf2[str(self.table)] * accuracy6)) 
                                   + np.square(accuracy5))
            self.ns_meas = self.cdf2['s_meas'].mean()
            self.n = len(self.cdf2['s_meas'])
            
            ### ISO 16000-8 option to calculate slope (defined to be calculated by Spread-Sheat/Excel)
            self.cdf2["runtime"] = np.arange(0,len(self.cdf2) * self.diff_sec, self.diff_sec)
            
            self.cdf2["t-te"] = self.cdf2["runtime"] - self.cdf2["runtime"][len(self.cdf2)-1]
            
            self.cdf2["lnte/t"] = self.cdf2["log"] - self.cdf2["log"][len(self.cdf2)-1]
            
            self.cdf2["slope"] = self.cdf2["lnte/t"] / self.cdf2["t-te"]
            
            try:
                if method=='iso':
                    self.slope = self.cdf2["slope"].mean()
                    
                    self.sumconz = self.cdf2["CO2_ppm"].iloc[1:-1].sum()
                    self.area_sup = (self.diff_sec * (self.cdf2[str(self.table)][0]/2 + self.sumconz + self.cdf2[str(self.table)][len(self.cdf2)-1]/2))
                    
                    self.cdf2.loc[[len(self.cdf2)-1], "slope"] = abs(self.slope)
                    
                    self.s_phi_e = self.cdf2["slope"][:-1].std()/abs(self.slope)
                    self.s_lambda = self.cdf2["slope"][:-1].std()/abs(self.cdf2["slope"][:-1].mean())
                    
                    print('ATTENTION: ISO 16000-8 method has a weak uncertainty evaluation consider using trapezoidal method is correcting this.')
                    
                elif method=='trapez':
                    ### More acurate option to calculate the solpe of each (sub-)curve
                    self.x1 = self.cdf2["runtime"].values
                    self.y1 = self.cdf2["log"].values
                    
                    from scipy.stats import linregress
                    self.slope = -linregress(self.x1,self.y1)[0]
                    self.reg_slope = linregress(self.x1,self.y1)
                    
                    self.cdf2.loc[[len(self.cdf2)-1], "slope"] = -self.reg_slope.slope
                    self.s_phi_e = (-self.cdf2["t-te"][len(self.cdf2)-1] * self.reg_slope.intercept_stderr)
                    self.s_lambda = (-self.cdf2["t-te"][len(self.cdf2)-1] * self.reg_slope.stderr)
                    
                    from numpy import trapz
                    self.area_sup = np.ptrapz(self.cdf2[str(self.table)].values, dx=self.diff_sec) # proof that both methods have same answer:  area_sup_2 = area_sup_1
                    
                    print('ATTENTION: Trapezoidal method is used in ISO 16000-8 and here also considered in the uncertainty evaluation. However, more precise results are given by applying the Simpson-Rule.')
                
                elif method=='simpson':
                    ### More acurate option to calculate the solpe of each (sub-)curve
                    self.x1 = self.cdf2["runtime"].values
                    self.y1 = self.cdf2["log"].values
                    
                    from scipy.stats import linregress
                    self.slope = -linregress(self.x1,self.y1)[0]
                    self.reg_slope = linregress(self.x1,self.y1)
                    
                    self.cdf2.loc[[len(self.cdf2)-1], "slope"] = -self.reg_slope.slope
                    self.s_phi_e = (-self.cdf2["t-te"][len(self.cdf2)-1] * self.reg_slope.intercept_stderr)
                    self.s_lambda = (-self.cdf2["t-te"][len(self.cdf2)-1] * self.reg_slope.stderr)
                    
                    from scipy.integrate import simpson
                    self.area_sup = sc.integrate.simpson(self.cdf2[str(self.table)].values, dx=self.diff_sec, even='first') # proof that both methods have same answer:  area_sup_2 = area_s
                
                else:
                    raise ResidenceTimeMethodError
                    
            except ResidenceTimeMethodError as err:
                print(err)
                        
            
            
            self.a_rest = self.cdf2[str(self.table)].iloc[-1]/abs(self.slope)
            self.a_tot = self.area_sup + self.a_rest
            
            self.tau = (self.area_sup + self.a_rest)/self.cdf2[str(self.table)][0]
                       
            try:  
                if method =='iso':
                    # Taken from DIN ISO 16000-8:2008-12, Equation D2 units are cm3.m-3.sec
                    self.sa_num = self.ns_meas * (self.diff_sec) * ((self.n - 1)/np.sqrt(self.n)) # Taken from DIN ISO 16000-8:2008-12, Equation D2 units are cm3.m-3.sec
                    
                    # The uncertainty of the summed trapezoidal method itself is not covered by ISO 16000-8.
                    self.sa_tm = 0
                    
                elif method =='trapez':
                    # Actually sa_num (the propagated uncertainty of the measurement) should be calculated this way
                    self.sa_num = (self.diff_sec) * self.ns_meas * np.sqrt((2*self.n -1)/2*self.n ) 
                    
                    # Aditionally the summed trapezoidal method itself has an uncertainty as well.
                    self.sa_tm = self.diff_sec**2/12*(self.cdf2["runtime"].loc[len(self.cdf2)-1]- self.cdf2["runtime"][0])*self.cdf2[str(self.table)][0]/self.tau**2
                
                elif method =='simpson':
                    # Actually sa_num (the propagated uncertainty of the measurement) should be calculated this way
                    self.sa_num = 1/3*self.diff_sec*self.ns_meas*np.sqrt(2+20*round(self.n/2-0.5))
                    
                    # Aditionally the summed trapezoidal method itself has an uncertainty as well.
                    self.sa_tm = self.diff_sec**4/2880*(self.cdf2["runtime"].loc[len(self.cdf2)-1]-self.cdf2["runtime"][0])*self.cdf2[str(self.table)][0]/self.tau**4
                
                else:
                    raise ResidenceTimeMethodError
                    
            except ResidenceTimeMethodError as err:
                print(err)
            
            
            
    
            self.s_rest = np.sqrt(pow(self.s_lambda,2) + pow(self.s_phi_e,2))
            self.sa_rest = self.s_rest * self.a_rest
            self.s_area = np.sqrt(pow(self.sa_num,2) + pow(self.sa_tm,2) + pow(self.sa_rest,2))/self.a_tot  # s_area is a relative uncertainty in percent
            self.s_total = np.sqrt(pow(self.s_area,2) + pow(0.05,2))
            self.s_total_abs = self.s_total * self.tau
                        
            self.tau_hr.append(self.tau/3600)
            self.cdf2["tau_hr"] = self.tau/3600
            self.cdf2.loc[:, "s_total"] = self.s_total
            self.cdf2.loc[:, "s_total_abs_hr"] = self.s_total_abs/3600
            self.s_total_abs_hr.append(self.s_total_abs/3600)

            self.df_tau.append(self.cdf2)
            
            self.cdf3 = self.cdf2.loc[:, ["datetime", str(self.table)]]
            self.cdf3 = self.cdf3.set_index("datetime")
            self.cdf_list.append(self.cdf3)
        
        self.mega_cdf = pd.concat(self.cdf_list,axis = 1).interpolate(method = "linear")
        # self.mega_cdf.columns = self.new_names
        self.mega_cdf["mean_delta"] = self.mega_cdf.mean(axis = 1)
        self.mega_cdf["std mean_delta"] = self.mega_cdf.std(axis = 1)
        # self.mega_cdf = self.mega_cdf.set_index("datetime")
        self.mega_cdf = self.mega_cdf.fillna(method="bfill", limit=2)
        self.mega_cdf = self.mega_cdf.fillna(method="pad", limit=2)
        
        self.tau_hr_mean = np.mean(self.tau_hr)
        self.s_tau_hr_mean = (np.sqrt(pow(np.array(self.s_total_abs_hr),2).sum()) 
                + statistics.variance(self.tau_hr))/len(np.array(self.tau_hr)-1)
        
    
        if plot:
            import plotly.io as pio
            
            pio.renderers.default='browser'
            pd.options.plotting.backend = "matplotlib"
            #######################################################################
            pd.options.plotting.backend = "plotly"
    
            import plotly.io as pio
            
            pio.renderers.default='browser'
            import plotly.express as px
            
           
            fig = px.line(self.mega_cdf, x=self.mega_cdf.index, y=self.mega_cdf.columns, title="mean of {}".format(self.experiment))
    
            fig.show()
            
            
            import plotly.io as pio
            
            pio.renderers.default='browser'
            pd.options.plotting.backend = "matplotlib"
            #self.df_tau, self.mega_cdfv
        return   [self.tau_hr_mean, self.s_tau_hr_mean], self.df_tau, self.mega_cdf
    
    def decay_curve_comparison_plot(self, save = False):
        """
        This method produces a plot that shows the decay curve of the selected
        experiment and corresponding curves if the experiment were to be a fully
        mixed ventilation or ideal plug flow ventilation. 
        
        Run this method to see the graph it will make more sense

        Parameters
        ----------
        save : BOOL, optional
            if True saves the plot to the default directory. The default is False.

        Returns
        -------
        figure
            returns a figure.

        """
        self.d = self.mean_curve()[2].loc[:,["mean_delta"]]
        self.d['mean_delta_norm'] = self.d["mean_delta"]/self.d["mean_delta"].iat[0]


        self.d["runtime"] = np.arange(0,len(self.d) * self.diff_sec, self.diff_sec)
        
        self.d["min"] = self.d["runtime"]/(np.mean(self.tau_nom) * 3600)
        self.d["min"] = 1 - self.d["min"]
        
        self.slope = 1/(np.mean(a.tau_hr) * 3600)
        
        
        self.fig, ax = plt.subplots()


        def func(x, a, b):
            return a * np.exp(-b * x)

        self.slope_50 = 1/(a.tau_nom *3600)
        y_50 = func(self.d["runtime"].values, 1, self.slope_50)
        self.d["ea_50"] = y_50
        self.d["ea_50_max"] = self.d[["min", "ea_50"]].max(axis = 1)
        
        self.d["mean_delta_norm_max"] = self.d[["min", "mean_delta_norm"]].max(axis = 1)
        
        
        
        ax.plot(self.d["runtime"], self.d["ea_50_max"].values, label = "50 % efficiency (estimated)")
        ax.plot(self.d["runtime"], self.d["mean_delta_norm_max"].values, label = "{} % efficiency (measured)".format(round(self.tau_nom/(np.mean(self.tau_hr)*2) * 100) ))
        ax.plot(self.d["runtime"], self.d["min"].values, label = "maximum effieiency (estimated)")
        
        
        
        ax.set_xlabel("time (sec)")
        ax.set_ylabel("CO2 (normalized)")
        
        ax.set_title("Decay curves for {}".format(self.experiment))
        ax.legend()

        if save:
            ax.savefig("{} decay curve comparison".format(self.experiment))
        
        return self.fig
        
    def outdoor_data(self):
        """
        This method calculates the mean , std, max and min of the parameters measured 
        on the outdoor of the measurement site. 
        The outdoor data comes from two sources. 1) from the HOBO sensor
        2) From the weather station

        Returns
        -------
        dataframe
            The dataframe contains the summary of the parameters for the selected
            experiment period

        """
        adf = pd.read_sql_query("SELECT * FROM weather.außen WHERE datetime BETWEEN '{}' AND '{}'".format(self.t0,self.tn), con = self.engine).drop("index", axis = 1).set_index("datetime")
        wdf = pd.read_sql_query("SELECT * FROM weather.weather_all WHERE datetime BETWEEN '{}' AND '{}'".format(self.t0,self.tn), con = self.engine).set_index("datetime")

        
        
        data = [
                [adf['temp_°C'].mean(), adf['temp_°C'].std(), adf['temp_°C'].max(), adf['temp_°C'].min()], 
                [adf['RH_%rH'].mean(), adf['RH_%rH'].std(), adf['RH_%rH'].max(), adf['RH_%rH'].min()],
                [self.aussen()["meanCO2"], self.Cout["sgm_CO2"], self.Cout["maxCO2"], self.Cout["minCO2"]],
                [wdf["Wind Speed, m/s"].mean(), wdf["Wind Speed, m/s"].std(), wdf["Wind Speed, m/s"].max(), wdf["Wind Speed, m/s"].min()],
                [wdf["Gust Speed, m/s"].mean(), wdf["Gust Speed, m/s"].std(), wdf["Gust Speed, m/s"].max(), wdf["Gust Speed, m/s"].min()],
                [wdf["Wind Direction"].mean(), wdf["Wind Direction"].std(), wdf["Wind Direction"].max(), wdf["Wind Direction"].min()],
                [wdf["Temperature °C"].mean(), wdf["Temperature °C"].std(), wdf["Temperature °C"].max(), wdf["Temperature °C"].min()],
                [wdf["RH %"].mean(), wdf["RH %"].std(), wdf["RH %"].max(), wdf["RH %"].min()]
                ]
       
        self.outdoor_summary = pd.DataFrame(data = data, index = ["temp_°C","RH_%rH", "CO2_ppm", "Wind Speed, m/s","Gust Speed, m/s","Wind Direction", "Temperature °C", "RH %" ], columns = ["mean", "std", "max", "min"] )
        
        
        return self.outdoor_summary
        
        
    def indoor_data(self):
        
        self.names = pd.read_sql_query('SHOW TABLES FROM {}'.format(self.database), con = self.engine)
        self.names = self.names.iloc[:,0].to_list()
        self.new_names = [x for x in self.names if (x not in self.exclude)]
        
        
        self.humidity = []
        self.temp = []
    
        for i in self.new_names:
            # print(i)
            self.hudf = pd.read_sql_query("SELECT * FROM {}.{} WHERE datetime BETWEEN '{}' AND '{}'".format(self.database,i,self.t0,self.tn), con = self.engine).set_index("datetime").dropna()
            if 'RH_%rH' in self.hudf.columns:
                self.humidity.append(self.hudf["RH_%rH"].mean())
            if 'temp_°C' in self.hudf.columns:
                self.temp.append(self.hudf["temp_°C"].mean())
        self.humidity = [x for x in self.humidity if x == x] 
        self.temp = [x for x in self.temp if x == x] # to remove nans
        self.indoor_list = [[statistics.mean(self.humidity), statistics.stdev(self.humidity), max(self.humidity), min(self.humidity)]]
        
        self.indoor_list.append([statistics.mean(self.temp), statistics.stdev(self.temp), max(self.temp), min(self.temp)])
        
        for i in self.testos:
            
            sdf = pd.read_sql_query("SELECT * FROM {}.{} WHERE datetime BETWEEN '{}' AND '{}'".format(self.database.lower(),i,self.t0,self.tn), con = self.engine)
            if not(sdf.empty):
                self.sdf = sdf.drop_duplicates(subset="datetime").set_index("datetime")
                self.sdf = self.sdf.loc[:,["hw_m/sec"]].dropna()
        self.indoor_list.append([self.sdf["hw_m/sec"].mean(), self.sdf["hw_m/sec"].std(), self.sdf["hw_m/sec"].max(), self.sdf["hw_m/sec"].min()])
        
        self.wadf = pd.read_sql_query("SELECT * FROM weather.{} WHERE datetime BETWEEN '{}' AND '{}'".format(self.wall_database,self.t0,self.tn), con = self.engine).set_index("datetime")

        self.indoor_list.append([self.wadf.mean().mean(), self.wadf.values.std(ddof=1), self.wadf.values.max(), self.wadf.values.min()])

        
        self.indoor_summary = pd.DataFrame(data = self.indoor_list, index = ["RH_%rH", "temp_°C", "hw_m/sec", "wall_temp_°C"], columns = ["mean", "std", "max", "min"] )

        return self.indoor_summary
        
    def outdoor_windspeed_plot(self, save = False):
        """
        This method produces a plot for the outdoor wind speeds during the measurement

        Parameters
        ----------
        save : BOOL, optional
            If True , the plot is saved. The default is False.

        Returns
        -------
        Figure.
            
        """
        global df1
        df1 = pd.read_sql_query("SELECT * FROM {}.{} WHERE datetime BETWEEN '{}' AND \
                            '{}'".format("weather", "weather_all", self.t0,\
                                self.tn), con = self.engine)
        df1 = df1.loc[:,['datetime', 'Wind Speed, m/s', 'Gust Speed, m/s', 'Wind Direction']]
        u = df1['Wind Direction'].to_numpy()
    
        U = np.sin(np.radians(u))
        V = np.cos(np.radians(u))
        wdf_plot = df1.set_index("datetime")
        wdf_plot['u'] = U
        wdf_plot['v'] = V
        wdf_plot['y'] = 0
    
    
        converter = mdates.ConciseDateConverter()
        munits.registry[np.datetime64] = converter
        munits.registry[datetime.date] = converter
        munits.registry[datetime.datetime] = converter
    
        fig, ax1 = plt.subplots()
        ax1.plot(wdf_plot['Gust Speed, m/s'],color = 'silver', label = 'Gust Speed', zorder=1)
    
    
    
        ax1.set_ylabel('Gust speed (m/sec)')
        ax1.set_xlabel('Time')
        # ax2 = ax1.twinx()
        ax1.plot(wdf_plot['Wind Speed, m/s'], label = 'Wind Speed', zorder=2)
        ax1.quiver(wdf_plot.index, wdf_plot['Wind Speed, m/s'], U,  V , width = 0.001, zorder=3)
        ax1.set_ylabel('wind speed (m/sec) and direction (up is north)')
    
        plt.ylim(bottom=-0.1)
        title = "Wind and Gust speed during {}".format(self.experiment)

        plt.legend( loc='upper right')
        plt.title(title)
        if save:
            plt.savefig(title + '.png', bbox_inches='tight', dpi=400)
        plt.show()     
       
        return fig
    
    def residence_time_sup_exh(self, experimentno=16, deviceno=0, periodtime=120, 
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
        self.alpha_mean, self.df_alpha, self.df_indoor = self.mean_curve()

        #%% Function to find outliers
        def find_outliers(col):
            from scipy import stats
            z = np.abs(stats.zscore(col))
            idx_outliers = np.where(z>3,True,False)
            return pd.Series(idx_outliers,index=col.index)
        
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
        
        time = pd.read_sql_query("SELECT * FROM testdb.timeframes;", con = self.engine)      
        #standard syntax to fetch a table from Mysql; In this case a table with the 
        # short-names of the measurements, all the start and end times, the DB-name 
        # of the measurement and the required table-names of the DB/schema is loaded into a dataframe. 
        
        # start, end = self.t0_20, self.tn_20
        
        fdelay = 2
        
        self.t0_2T = time.loc[time['short_name']==self.experiment].iat[0,3] + dt.timedelta(seconds=fdelay*T)                                 
        # actual start of the experiment, out of the dataframe "time" + device periods,
        # since the decay of the moving average curve of the subsystem 23 is at the 
        # beginning falsified by the drop from the accumulation level in subsystem 3.
        # The drop is the response signal of the entire system 123. After this
        # about 2*T the falisification due to the respones of the entire system is
        # negligable.
        
        # table = time["tables"][t].split(",")[l]                                         #Name of the ventilation device
        
        # dum = [["Experiment",time["short_name"][t] ], ["Sensor", table]]                # Creates a list of 2 rows filled with string tuples specifying the experiment and the sensor.
        # if experimentname:
        #     print(tabulate(dum))                                                            # Prints the inut details in a table
        # else:
        #     pass
        
        database = self.database                                                 # Selects the name of the database as a string 
        
        #%%% Load data for the occupied space V3
        
        #experimentglo = CBO_ESHL(experiment = dum[0][1], sensor_name = dum[1][1])
        
        alpha_mean_u = ufloat(self.alpha_mean[0], self.alpha_mean[1])
        
        self.dfin_dCmean = self.df_indoor.loc[:,['mean_delta', 'std mean_delta']]
        
        while not(self.t0 in self.dfin_dCmean.index.to_list()):
                    self.t0 = self.t0 + dt.timedelta(seconds=1)
                    # print(self.t0)
        
        mean_delta_0_room = self.dfin_dCmean.loc[self.t0]
        dfin_dCmean = self.dfin_dCmean.copy()
        mean_delta_0_room_u = ufloat(mean_delta_0_room[0],mean_delta_0_room[1])
           
        #%%%%% Add mean and exhaust concentrations indoor (V3) to the dfin_dCmean
        '''
            mean concentrations: 
                Based on the calculated spatial and statistical mean air
                age in the occupied space and the spacial average initial 
                concentration in the occupied space at t0glob.
        '''
        
        count = 0
        dfin_dCmean['room_av'] = pd.Series(dtype='float64')
        dfin_dCmean['std room_av'] = pd.Series(dtype='float64')
        dfin_dCmean['room_exh'] = pd.Series(dtype='float64')
        dfin_dCmean['std room_exh'] = pd.Series(dtype='float64')
        dfin_dCmean.reset_index(inplace=True)
        
        while (count < len(dfin_dCmean)):     
            
            '''
            mean concentrations: 
                Based on the calculated spatial and statistical mean air
                age in the occupied space and the spacial average initial 
                concentration in the occupied space at t0glob.
            '''    
            value = mean_delta_0_room_u*unumpy.exp(-1/(alpha_mean_u)*\
                        ((dfin_dCmean['datetime'][count]-self.t0).total_seconds()/3600))
            dfin_dCmean['room_av'][count] = value.n
            dfin_dCmean['std room_av'][count] = value.s
            
            '''
            exhaust concentrations: 
                Based on the calculated spatial and statistical mean air
                age in the occupied space and the spacial average initial 
                concentration in the occupied space at t0glob.
            '''
            value = mean_delta_0_room_u*unumpy.exp(-1/(2*alpha_mean_u)*\
                        ((dfin_dCmean['datetime'][count]-self.t0).total_seconds()/3600))
            dfin_dCmean['room_exh'][count] = value.n
            dfin_dCmean['std room_exh'][count] = value.s
            
            count = count + 1
        
        dfin_dCmean = dfin_dCmean.set_index('datetime')           
            
        #%%% Load background data
           
        background = self.aussen()['meanCO2']                                       # Future: implement cyclewise background concentration; Till now it takes the mean outdoor concentration of the whole experiment.
        
        #%%% Load data of the experiment and the selected sensor
        
        df = pd.read_sql_query("SELECT * FROM {}.{} WHERE datetime BETWEEN '{}' AND\
                               '{}'".format(self.database, self.aperture_sensor, self.t0_20, self.tn_20), con = self.engine)
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
        
        while not(self.t0_2T in df.index.to_list()):                                            # The t0 from the excel sheet may not be precice that the sensor starts 
            self.t0_2T = self.t0_2T + dt.timedelta(seconds=1)                                           # - starts at the same time so i used this while loop to calculate the 
            print(self.t0_2T)                                                                   # - the closest t0 after the original t0
        
        df["roll"] = df["CO2_ppm"].rolling(int(T/diff)).mean()                          # moving average for 2 minutes, used to calculate Cend; T = 120s is the period time of the push-pull ventilation devices which compose the ventilation system. 
        
        
        
        c0 = df["roll"].loc[self.t0_2T]                                                      # C0; @DRK: Check if c0 = df["roll"].loc[t0] is better here. ## ORIGINAL: c0 = df["CO2_ppm"].loc[t0] 
        Cend37 = round((c0)*0.37, 2)                                                    # @DRK: From this line 101 schould be changed.   
        
        cend = df.loc[df["roll"].le(Cend37)]                                            # Cend: Sliced df of the part of the decay curve below the 37 percent limit
        
        if len(cend) == 0:                                                              # Syntax to find the tn of the experiment
            self.tn = str(df.index[-1])
            print("The device has not reached 37% of its initial concentration")
        else:
            self.tn = str(cend.index[0])
        
           
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
        
        sup_exh_df = pd.concat([dfin_dCmean, df_sup2, df_exh2], axis = 1).reset_index()
        sup_exh_df.columns = ["datetime", 
                                  "meas room_av", "std meas room_av", 
                                  "calc room_av", "std calc room_av",  
                                  "calc room_exh", "std calc room_exh", 
                                  "supply", "exhaust"]
        sup_exh_df['d calc exh-av'] = np.sqrt(np.power(sup_exh_df["calc room_exh"],2)\
                                              - np.power(sup_exh_df["calc room_av"],2))
        sup_exh_df['std d calc exh-av'] = np.sqrt(np.power(sup_exh_df["std calc room_exh"],2)\
                                                  - np.power(sup_exh_df["std calc room_av"],2))
        
        #%%%%%% Calculation of the weight factor of the current device period
                
        ddCmax_exhav = sup_exh_df.loc[sup_exh_df['d calc exh-av'].idxmax()]
        ddCmax_exhav = ddCmax_exhav.filter(['datetime','d calc exh-av','std d calc exh-av'])
        
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
        
        start_date = str(self.t0_2T); end_date = self.tn # CHANGE HERE 
        
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
        for i in range(1, int(df_sup3.num.max()+1)):
        
            try:
                if export_sublist and len(df_sup3.loc[df_sup3["num"]==i]) > 3:
                    dummy_df = dummy_df.append(df_sup3.loc[df_sup3["num"]==(i)])
                    dummy_df = dummy_df.rename(columns = {'CO2_ppm':'CO2_ppm_{}'.format(i)})
                
                
            except KeyError:
                pass
                # print("ignore the key error")
        
            df_sup_list.append(df_sup3.loc[df_sup3["num"]==i])
        
        del dummy_df["num"]
        dummy_df.to_csv(r'D:\Users\sauerswa\wichtige Ordner\sauerswa\Codes\Python\Recirculation\export\df_sup_{}_{}.csv'.format(self.database, self.aperture_sensor), index=True) 
        
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
        
        res = reg_result[reg_result['sensor'].str.lower() == self.aperture_sensor].reset_index(drop = True) # Contains the sensor calibration data and especially the calibration curve.
        accuracy1 = 50 # it comes from the equation of uncertainity for testo 450 XL
        accuracy2 = 0.02 # ±(50 ppm CO2 ±2% of mv)(0 to 5000 ppm CO2 )
                
        accuracy3 = 50 # the same equation for second testo 450 XL
        accuracy4 = 0.02
                
        accuracy5 = 75 # # the same equation for second tes
        accuracy6 = 0.03 # Citavi Title: Testo AG
        
        df_tau_sup = []
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
                    s_total = np.sqrt(pow(s_area,2) + pow(0.05,2))
                    
                    a.loc[:, "s_total"] = s_total
                    
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
            
        self.df_tau_s = pd.DataFrame({'Cycle':cyclnr_sup,
                                 'tau_sup':tau_list_sup, 
                                 'std tau_sup':stot_list_sup, 
                                 'weight':weight_list_sup, 
                                 'std weight':saw_list_sup})
        
        # Filter outliers (see https://medium.com/@stevenewmanphotography/eliminating-outliers-in-python-with-z-scores-dd72ca5d4ead)
        self.df_tau_s['outliers'] = find_outliers(self.df_tau_s['tau_sup'])
        self.df_tau_s = self.df_tau_s[self.df_tau_s['outliers']==False]
        
        
        '''
        Weighting factor for the supply phases is not as important since the 
        residence times here are mostly normal distributed throughout the phases.
        Therefore it can be set low which means that almost all calcuated 
        residence times will be considered. The range for cfac_s is 0 to 1.
        Values >= 1 will autmatically trigger that the residence times with the 
        highest weighting factor will be chosen.
        '''
        cfac_s = 0.2
        self.df_tau_s2 = self.df_tau_s[self.df_tau_s['weight']>cfac_s]
        if len(self.df_tau_s2) == 0:
            self.df_tau_s2 = self.df_tau_s.nlargest(10, 'weight')
        
        tau_list_sup_u = unumpy.uarray(self.df_tau_s2['tau_sup'],self.df_tau_s2['std tau_sup'])
        self.weight_list_sup_u = unumpy.uarray(self.df_tau_s2['tau_sup'],self.df_tau_s2['std tau_sup']) 
        
        # Mean supply phase residence time
        self.tau_s_u = sum(tau_list_sup_u*self.weight_list_sup_u)/sum(self.weight_list_sup_u)
            
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
                                     x = self.df_tau_s['Cycle'], 
                                     y = self.df_tau_s['tau_sup'],
                                     error_y=dict(value=self.df_tau_s['std tau_sup'].max())
                                     ),
                           secondary_y=False,
                           )
            
            fig2.add_trace(go.Scatter(name='Gewichtung',
                                     x = self.df_tau_s['Cycle'], 
                                     y = self.df_tau_s['weight'],
                                     error_y=dict(value=self.df_tau_s['std weight'].max())
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
        dummy_df.to_csv(r'D:\Users\sauerswa\wichtige Ordner\sauerswa\Codes\Python\Recirculation\export\df_exh_{}_{}.csv'.format(self.database, self.aperture_sensor), index=True) 
            
            
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
                    #         concentration in the occupied space at t0glob.
                    # '''
                    
                    # count = 0
                    # b['room_av'] = pd.Series(dtype='float64')
                    # b['std room_av'] = pd.Series(dtype='float64')
                    # while (count < len(b['datetime'])):     
                    #     value = mean_delta_0_room_u*unumpy.exp(-1/(alpha_mean_u)*((b['datetime'][count]-t0glob).total_seconds()/3600))
                    #     b['room_av'][count] = value.n
                    #     b['std room_av'][count] = value.s
                    #     count = count + 1            
                    
                    #%%%%% Add exhaust concentrations indoor (V3) to the exhaust dataframe
                    '''
                        exhaust concentrations: 
                            Based on the calculated spatial and statistical mean air
                            age in the occupied space and the spacial average initial 
                            concentration in the occupied space at t0glob.
                    '''
                    
                    count = 0
                    b['room_exh'] = pd.Series(dtype='float64')
                    b['std room_exh'] = pd.Series(dtype='float64')
                    while (count < len(b['datetime'])):     
                        value = mean_delta_0_room_u*unumpy.exp(-1/(2*alpha_mean_u)*\
                                    ((b['datetime'][count]-self.t0).total_seconds()/3600))
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
                    s_total = np.sqrt(pow(s_area,2) + pow(dC3e.s/dC3e.n,2))
                    
                    b.loc[:, "s_total"] = s_total
                    
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
            
        self.df_tau_e = pd.DataFrame({'Cycle':cyclnr_exh,
                                 'tau_exh':tau_list_exh, 
                                 'std tau_exh':stot_list_exh, 
                                 'weight':weight_list_exh, 
                                 'std weight':saw_list_exh})
        
        # Filter outliers (see https://medium.com/@stevenewmanphotography/eliminating-outliers-in-python-with-z-scores-dd72ca5d4ead)
        self.df_tau_e['outliers'] = find_outliers(self.df_tau_e['tau_exh'])
        self.df_tau_e = self.df_tau_e[self.df_tau_e['outliers']==False]
        
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
        df_tau_e2 = self.df_tau_e[self.df_tau_e['weight']>cfac_e]
        if len(df_tau_e2) == 0:
            df_tau_e2 = self.df_tau_e.nlargest(10, 'weight')
        
        tau_list_exh_u = unumpy.uarray(df_tau_e2['tau_exh'],df_tau_e2['std tau_exh'])
        weight_list_exh_u = unumpy.uarray(df_tau_e2['tau_exh'],df_tau_e2['std tau_exh']) 
        
        self.tau_e_u = sum(tau_list_exh_u*weight_list_exh_u)/sum(weight_list_exh_u)
        
        #%%%%% Plot: residence times of the step-up curves during exhaust-phase  
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
            fig = make_subplots(specs=[[{"secondary_y": True}]])
                           
            fig.add_trace(go.Scatter(name='Verweilzeit',
                                     x = self.df_tau_e['Cycle'], 
                                     y = self.df_tau_e['tau_exh'],
                                     error_y=dict(value=self.df_tau_e['std tau_exh'].max())
                                     ),
                           secondary_y=False,
                           )
            
            
            fig.add_trace(go.Scatter(name='Gewichtung',
                                     x = self.df_tau_e['Cycle'], 
                                     y = self.df_tau_e['weight'],
                                     error_y=dict(value=self.df_tau_e["std weight"].max())
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
        
        #%% returned values
        """
            Returns:
                self.t0 = initial timestamp of the start of the experiment
                self.tn = final timestamp of the evaluated data
                tau_e = exhaust residence time of the short-cut volume
                tau_s = exhaust residence time of the recirculation volume 
                        ("supply residence time")
        """
        
        return [self.t0, self.tn, self.tau_e_u, self.df_tau_e, self.tau_s_u, self.df_tau_s]      
        
        
#%% Execution of the Class
""" Inputs needed: 
    1) experiment name (look master_time_sheet.xlsx) 
    2) The name of the sensor (1a_testo, 2a_testo, 3a_testo) 
    
    If data is missing probably it is not available, 
    eitherways if you miss something please notify me
    Cheers
"""

a = CBO_ESHL("W_H_e1_Herdern" )
m=a.residence_time_sup_exh()
# t0, tn, tau_e_u, df_tau_e, tau_s_u, df_tau_s = a.residence_time_sup_exh()
