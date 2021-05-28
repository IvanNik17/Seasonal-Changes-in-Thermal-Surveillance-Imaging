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

from sunrise_sunset import sun



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
        
    
    metadata["Day/Night"] = day_night
    
    return metadata


if __name__ == '__main__':

    metadata_path = r"splits\feb_week.csv"
    metadata = pd.read_csv(metadata_path)
    metadata["DateTime"] = pd.to_datetime(metadata['DateTime'], dayfirst = True)
    
    metadata = add_day_night(metadata)
    
    
    
    
    
    
    
    
    
    
    