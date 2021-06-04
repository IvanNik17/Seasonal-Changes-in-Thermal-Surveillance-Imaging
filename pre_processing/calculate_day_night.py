# -*- coding: utf-8 -*-
"""
Created on Fri May 28 13:39:23 2021

@author: IvanTower
"""
import numpy as np
import pandas as pd
import datetime
from dateutil import tz

import os

from pre_processing.sunrise_sunset import sun

from pre_processing.sunposition import sunpos


def calculate_sunposition(metadata, lat = 57.0488, long=9.9217, local_zone = 'Europe/Copenhagen'):
    
    sunpos_az = []
    sunpos_zen = []
    for curr_time in metadata["DateTime"]:
    
          # utc_zone = tz.gettz('UTC')
          # local_zone = tz.gettz(local_zone)
          
          # curr_time = curr_time.astimezone(local_zone)
          # curr_time_utc = curr_time.astimezone(utc_zone)

          az,zen = sunpos(curr_time,lat,long,0)[:2]
          sunpos_az.append(az)
          sunpos_zen.append(zen)
        
          print(f"time {curr_time} has been processed - {az}, {zen}")
    
    metadata["SunPos_azimuth"] = sunpos_az
    metadata["SunPos_zenith"] = sunpos_zen
    
    return metadata
    



def add_day_night(metadata, lat = 57.0488, long=9.9217, local_zone = 'Europe/Copenhagen'):
    
    day_night = []
    for curr_time in metadata["DateTime"]:
    
        s=sun(lat=lat, long = long, local_zone = local_zone)
        
        sunrise_datetime = s.sunrise(when=curr_time)
        sunset_datetime = s.sunset(when=curr_time)
        
        curr_time = curr_time.replace(tzinfo=tz.gettz(local_zone))
        
        if curr_time>sunrise_datetime and curr_time<sunset_datetime:
            day_night.append("day")
            print(f"sunrise:{sunrise_datetime.time()}, sunset:{sunset_datetime.time()}  {curr_time} it's day")
        else:
            day_night.append("night")
            print(f"sunrise:{sunrise_datetime.time()}, sunset:{sunset_datetime.time()} {curr_time} it's night")
        
    
    metadata["Day_Night"] = day_night
    
    return metadata


if __name__ == '__main__':

    metadata_path = r"D:\2021_workstuff\Seasonal-Changes-in-Thermal-Surveillance-Imaging\splits\feb_week.csv"
    metadata = pd.read_csv(metadata_path)
    metadata["DateTime"] = pd.to_datetime(metadata['DateTime'], dayfirst = True)
    
    metadata = add_day_night(metadata)
    
    metadata = calculate_sunposition(metadata)
    
    
    
    
    
    
    
    
    
    
    