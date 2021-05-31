# -*- coding: utf-8 -*-
"""
Created on Mon May 31 08:05:44 2021

@author: IvanTower
"""

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

import os
import sys

path = os.path.join(os.path.dirname(__file__), os.pardir)

if path not in sys.path:
    sys.path.append(path)
    
from pre_processing.calculate_day_night import add_day_night


def plot_error_vs_other(df, columns,save_dir):
    #sns.scatterplot(data=df, x="DateTime", y="MSE", hue="model")
    
    f, ax = plt.subplots(figsize=(10, 5))
    pl = sns.lineplot(ax = ax, data=df, x=columns[0], y=columns[1], hue="model")

    pl.xaxis.set_major_locator(ticker.MultipleLocator(15))
    plt.xticks(rotation=45)

    plt.show()
    
    save_dir = os.path.join(save_dir,f'plot_{columns[0]}_{columns[1]}.png')
    
    plt.savefig(save_dir)
    
    
    
def calculate_correlation_mat(df,save_dir):
    f, ax = plt.subplots(figsize=(10, 10))
    
    df = df.drop(['Folder name', 'Image Number'], axis=1)
    
    heatmap = sns.heatmap(df.corr(),ax = ax, vmin=-1, vmax=1, annot=True)
    heatmap.set_title("Correlation Matrix")
    plt.show()
    
    save_dir = os.path.join(save_dir,'correlation_matrix.png')
    
    plt.savefig(save_dir)
    
    
def calculate_depended_value_corr(df,save_dir):
    f, ax = plt.subplots(figsize=(8, 12))
    
    df = df.drop(['Folder name', 'Image Number'], axis=1)
    
    heatmap = sns.heatmap(df.corr()[['MSE']].abs().sort_values(by='MSE', ascending=False),ax = ax, vmin=0, vmax=1, annot=True, cmap='BrBG')
    heatmap.set_title('Features Correlating with MSE', fontdict={'fontsize':18}, pad=16)
    
    save_dir = os.path.join(save_dir,'dependent_value_correlation.png')
    
    plt.savefig(save_dir)
    
    
def plot_barplots(df, columns,save_dir):
    
    f, ax = plt.subplots(figsize=(10, 5))
    bp = sns.barplot(ax = ax, x=columns[0], y=columns[1], data=df, hue="model")
    
    plt.xticks(rotation=45)

    plt.show()
    
    save_dir = os.path.join(save_dir,f'barplot_{columns[0]}_{columns[1]}.png')
    
    plt.savefig(save_dir)
    

def augment_dataframe(df):
    df = add_day_night(df)
    
    df["Hour"] = df["DateTime"].dt.hour
    
    
    weekdays_coding = {
      0: "Monday",
      1: "Tuesday",
      2: "Wednesday",
      3: "Thursday",
      4: "Friday",
      5: "Saturday",
      6: "Sunday"
    }
    
    df['Weekday'] = df['DateTime'].dt.weekday
    df['Weekday_name'] = list(map(weekdays_coding.get, df['Weekday']))
    
    timeslots_times = np.array([0,3,6,9,12,15,18,21])

    timeslots_coding = {
      1: "Early Post-Midnight",
      2: "Late Post-Midnight",
      3: "Early Morning",
      4: "Late Morning",
      5: "Early Afternoon",
      6: "Late Afternoon",
      7: "Early Evening",
      8: "Late Evening",
    }
    
    curr_timeslots = np.searchsorted(timeslots_times, df['DateTime'].dt.hour, side='right')
    df['Timeslot'] = curr_timeslots
    
    df['Timeslot_name'] = list(map(timeslots_coding.get, df['Timeslot']))
    
    
    
    return df

if __name__ == '__main__':
    
    method_folder = "VQVAE"
    
    models = ['feb_day', 'feb_week', 'feb_month']
    months = ['results_jan', 'results_apr', 'results_aug']

    df = None
    for model in models:

        for month in months:
            path = os.path.join(method_folder,model,month+'.csv')

            df_ = pd.read_csv(path)
            # df_["DateTime"] = pd.to_datetime(df_['DateTime'], dayfirst = True).dt.strftime('%m-%d')
            df_["DateTime"] = pd.to_datetime(df_['DateTime'], dayfirst = True)

            df_['model'] = model
            df_['month'] = month

            if df is None:
                df = df_
            else:
                df = df.append(df_)
                
                
    save_folder = os.path.join(method_folder,"results")
    if not os.path.exists(save_folder):

        os.makedirs(save_folder)
                
   
    df = augment_dataframe(df)
    
    calculate_correlation_mat(df,save_folder)
                
    calculate_depended_value_corr(df, save_folder)
    
    plot_error_vs_other(df,["Temperature", "MSE"],save_folder)
    
    plot_error_vs_other(df,["Humidity", "MSE"],save_folder)
             


    plot_barplots(df, ["Hour", "MSE"], save_folder)    
    
    plot_barplots(df, ["Day_Night", "MSE"], save_folder)   
    
    plot_barplots(df, ["Timeslot_name", "MSE"], save_folder)
    
    plot_barplots(df, ["Weekday_name", "MSE"], save_folder)   
                