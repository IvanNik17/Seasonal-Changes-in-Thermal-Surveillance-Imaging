# -*- coding: utf-8 -*-
"""
Created on Fri Jun  4 11:37:31 2021

@author: IvanTower
"""

import pandas as pd
import numpy as np
import os
import sys

path = os.path.join(os.path.dirname(__file__), os.pardir)

if path not in sys.path:
    sys.path.append(path)
    
from pre_processing.calculate_day_night import add_day_night, calculate_sunposition
from pre_processing.activity_calculation import augment_activity
from pre_processing.count_people import count_people_annotations


def augment_dataframe(df):
    
    mask_path = r"D:\2021_workstuff\Seasonal-Changes-in-Thermal-Surveillance-Imaging\pre_processing\mask_ropes_water.png"
    images_path = r"E:\0_Monthly_videoPipeline\output_image_v3"
    
    df_list = []
    for curr_model in df['model'].unique():
        df_small = df[df['model'] == curr_model]
    
        df_small = df_small.iloc[: , :-3]
    
        cfg = {
            'image_path': images_path,
            'metadata': df_small,
            'mask_path': mask_path
        }
        
        df_small = augment_activity(cfg)
        
        df_list.append(df_small)
    
    activity_column = pd.concat(df_list)['Activity']
    
    df['Activity'] = activity_column
    
    
    df = count_people_annotations(df)
    
    
    
    df = add_day_night(df)
    
    
    df = calculate_sunposition(df)
    
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