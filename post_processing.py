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

        

    def volume(self):
        try:
            if 'cbo' in self.database:
                self.v = pd.DataFrame([{'Volume V23 in m³':195.1, 'std Volume V23 in m³':195.1*0.1}])
            elif 'eshl' in self.database:
                self.v = pd.DataFrame([{'Volume V23 in m³':133.9, 'std Volume V23 in m³':133.9*0.1}])
            else:
                string = 'The current CBO_ESHL.database is not valid. A volume can not be returned CBO_ESHL.volume().'
                raise NameError
                pass
        except NameError:
            prYellow(string)
        return self.v
        
    def volume_flow(self,  level_eshl = "WZ_100", level_cbo = "K1_St5" ):
        self.vf = self.times[self.times["experiment"] == self.experiment].iloc[0,7]
        self.vf_std = self.times[self.times["experiment"] == self.experiment].iloc[0,8]
        
        try:
            if "eshl" in self.database:
                sheet = "ESHL_Vdot"
                self.level = level_eshl
                df = pd.read_excel("Vdot_results.xlsx", sheet_name = sheet)
                self.vdot_sup = df.loc[df["Level"] == self.level].iat[0,1]
                self.vdot_sup_std = df.loc[df["Level"] == self.level].iat[0,2]
                
                self.vdot_exh = abs(df.loc[df["Level"] == self.level].iat[0,3])
                self.vdot_exh_std = df.loc[df["Level"] == self.level].iat[0,4]
                print(self.level)
            elif 'cbo' in self.database:
                sheet = "CBo_Vdot"
                self.level = level_cbo
    
                df = pd.read_excel("Vdot_results.xlsx", sheet_name = sheet)
                self.vdot_sup = df.loc[df["Level"] == self.level].iat[0,1]
                self.vdot_sup_std = df.loc[df["Level"] == self.level].iat[0,2]
                
                self.vdot_exh = abs(df.loc[df["Level"] == self.level].iat[0,3])
                self.vdot_exh_std = df.loc[df["Level"] == self.level].iat[0,4]
                print(self.level)
            
            else:
                raise NameError
                pass
        except NameError:
               return 'The current CBO_ESHL.database is not valid. Volumeflows can not be returned CBO_ESHL.volume_flow().'

        
        data = [[self.experiment, self.vf, self.vf_std, self.level , self.vdot_sup, self.vdot_sup_std, self.vdot_exh, self.vdot_exh_std]]
        columns = ["experiment", "volume_flow", "volume_flow_std","level", "vdot_sup", "vdot_sup_std", "vdot_exh", "vdot_exh_std"]
        index = ["0"]
        self.volume_flows = pd.DataFrame(data, columns=columns, index = index)
        
        return self.volume_flows
    
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
        
#%% Execution of the Class
""" Inputs needed: 
    1) experiment name (look master_time_sheet.xlsx) 
    2) The name of the sensor (1a_testo, 2a_testo, 3a_testo) 
    
    If data is missing probably it is not available, 
    eitherways if you miss something please notify me
    Cheers
"""

a = CBO_ESHL("W_H_e1_Herdern" )
# m=a.residence_time_sup_exh()
# t0, tn, tau_e_u, df_tau_e, tau_s_u, df_tau_s = a.residence_time_sup_exh()
