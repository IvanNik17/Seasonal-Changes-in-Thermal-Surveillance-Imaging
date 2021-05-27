# -*- coding: utf-8 -*-
"""
Created on Wed May 19 09:47:33 2021

@author: Ivan
"""
import numpy as np
import pandas as pd
import os

from scipy.spatial.distance import pdist, squareform


import matplotlib.pyplot as plt

# furthest point sampling
def getGreedyPerm(x,n_points=1024):
    """
    A Naive O(N^2) algorithm to do furthest points sampling
    Parameters
    ----------
    x : ndarray (N, N)
        An NxN distance matrix for points
    Return
    ------
    tuple (list, list)
        (permutation (N-length array of indices),
        lambdas (N-length array of insertion radii))
    """
    N = x.shape[0]
    #print(x.shape)
    #By default, takes the first point in the list to be the
    #first point in the permutation, but could be random
    perm = np.zeros(N, dtype=np.int64)
    lambdas = np.zeros(N)
    ds = x[0, :]
    for i in range(1, N):
        idx = np.argmax(ds)
        perm[i] = idx
        lambdas[i] = ds[idx]
        ds = np.minimum(ds, x[idx, :])
    #print(perm[:])
    return perm[:n_points]

# Function to get data for coldest, hottest and median months
# Add the full metadata and select - min for coldest, max for hottest, median for median one


def get_specific(metadata):
    metadata =  metadata.set_index('DateTime')

    metadata_grouped = metadata.groupby(pd.Grouper(freq='M')).median()
    result = metadata_grouped[metadata_grouped["Temperature"] == metadata_grouped["Temperature"].median()]

    return result


def get_experiment_data_v2(metadata,which = "min"):


    if which == "min":

        compare_month = metadata.groupby([metadata["DateTime"].dt.month]).min()
        month= compare_month.index[compare_month["Temperature"] == compare_month ["Temperature"].min()][0]

        needed_month = metadata[metadata["DateTime"].dt.month == month]

        compare_day = needed_month.groupby([needed_month["DateTime"].dt.day]).min()
        day = compare_day.index[compare_day['Temperature']==compare_day['Temperature'].min()][0]


    elif which =="max":

        compare_month = metadata.groupby([metadata["DateTime"].dt.month]).max()
        month= compare_month.index[compare_month["Temperature"] == compare_month ["Temperature"].max()][0]

        needed_month = metadata[metadata["DateTime"].dt.month == month]

        compare_day = needed_month.groupby([needed_month["DateTime"].dt.day]).max()
        day = compare_day.index[compare_day['Temperature']==compare_day['Temperature'].max()][0]

    elif which == "median":

        compare_month = metadata.groupby([metadata["DateTime"].dt.month]).quantile(interpolation='nearest')
        month= compare_month.index[compare_month["Temperature"] == compare_month ["Temperature"].quantile(interpolation='nearest')][0]

        needed_month = metadata[metadata["DateTime"].dt.month == month]

        compare_day = needed_month.groupby([needed_month["DateTime"].dt.day]).quantile(interpolation='nearest')
        day = compare_day.index[compare_day['Temperature']==compare_day['Temperature'].quantile(interpolation='nearest')][0]



    needed_day = needed_month[needed_month["DateTime"].dt.day == day]

    all_possible_days = needed_month["DateTime"].dt.day.unique()
    dist = np.abs(all_possible_days - day)
    ind= np.argsort(dist)[:7]
    weekdays = all_possible_days[ind]

    needed_week = needed_month[needed_month["DateTime"].dt.day.isin(weekdays)]


    # reset indices to start from 0
    needed_day = needed_day.reset_index(drop=True)
    needed_week = needed_week.reset_index(drop=True)
    needed_month = needed_month.reset_index(drop=True)

    return needed_day, needed_week, needed_month



def get_experiment_data(metadata, which = "min"):

    mask = []

    if which == "min":
        mask = metadata["Temperature"] == metadata["Temperature"].min()
    elif which =="max":
        mask = metadata["Temperature"] == metadata["Temperature"].max()
    elif which == "median":
        mask = metadata["Temperature"] == metadata["Temperature"].median()

    # get the day number based on mask - here using "Folder name", as it is the date
    needed_day_number = metadata[mask].iloc[0]["Folder name"]
    # get all rows that have the same Folder name
    needed_day_frames = metadata[metadata["Folder name"] == needed_day_number]
    num_frames_needed_day = len(needed_day_frames)

    # get the week number using the isocalendar().week, on the DateTime column
    needed_week_number = metadata[mask]["DateTime"].dt.isocalendar().week.iloc[0]
    #get all rows that have the same week number
    needed_week_frames = metadata[metadata["DateTime"].dt.isocalendar().week == needed_week_number]
    num_frames_needed_week = len(needed_week_frames)

    # get the month number using the month call on the DateTime column
    needed_month_number = metadata[mask]["DateTime"].dt.month.iloc[0]
    #  get all rows that have the same month number
    needed_month_frames = metadata[metadata["DateTime"].dt.month == needed_month_number]
    num_frames_needed_month = len(needed_month_frames)

    # reset indices to start from 0
    needed_day_frames = needed_day_frames.reset_index(drop=True)
    needed_week_frames = needed_week_frames.reset_index(drop=True)
    needed_month_frames = needed_month_frames.reset_index(drop=True)

    return needed_day_frames, needed_week_frames, needed_month_frames


def extract_furthest_frames_metadata_daily(metadata, features, num_frames = 100, reserved_at_ends = 5, debug = False):
    """
    Function that takes the day, week, month sub-datasets,
    the features that are needed and gets farthest point frames
    ----------
    metadata : pandas dataframe, one day worth of data

    features : columns from the metadata dataframe to be used in the distance calculation
        example ["Temperature"] or ["Temperature", "Humidity"]

    num_frames : how many frames to be extracted from the furthest point sampling

    reserved_at_ends : how many points at both ends of each clip to be reserved and not sampled

    Return
    ------
    pandas dataframe containing the sampled frame metadata for the current day
    """

    # Add the "Image Number" to the features
    curr_features = features.copy()
    curr_features.append('Image Number')
    # Make a sub-dataframe for the calculations
    metadata_small = metadata[curr_features]

    metadata_small = metadata_small.reset_index(drop=True)

    # make a new column containing the indices, before cutting the reserved ones
    metadata_small["Numbering"] = metadata_small.index

    # get all the unique Image Numbers
    unique_frame_nums = metadata_small['Image Number'].unique()

    # get the reserved_at_ends from the front and back of the array
    reserved_frame_nums = unique_frame_nums[:reserved_at_ends]
    reserved_frame_nums = np.concatenate((reserved_frame_nums, unique_frame_nums[-reserved_at_ends:]), axis=0)

    # remove them from the sub-dataframe
    metadata_small = metadata_small[(~metadata_small["Image Number"].isin(reserved_frame_nums))]

    # save the numbering before normalization
    indices_before_cutting = metadata_small["Numbering"]

    #  remove the Image Number column as it is not needed for the pdist calc
    metadata_small = metadata_small.drop(["Image Number"], axis=1)

    # normalize the columns between the minimum and maximum
    for column in metadata_small.columns:
        metadata_small[column] = (metadata_small[column] - np.min(metadata_small[column]))/np.ptp(metadata_small[column])

    # use the scipy pdist to calculate the distances between all elements
    metadata_small_dist = pdist(metadata_small)
    #  make the square distance matrix
    metadata_small_dist_square = squareform(metadata_small_dist)

    #  use the greedyPerm to get the sampled frame indices
    indices_search = getGreedyPerm(metadata_small_dist_square, num_frames)

    #  get the indices in the context of the metadata
    indices_metadata = indices_before_cutting.iloc[indices_search].to_numpy()

    # get the sampled frame metadata

    if debug:
        return metadata.iloc[indices_metadata],metadata_small, indices_search
    else:
        return metadata.iloc[indices_metadata]

#  Function that separates the required samplings per day depending on how many days the dataset contains
def extract_frames_metadata(metadata, features, num_frames = 100, reserved_at_ends = 5, debug = False):
    """
    Function that separates the required samplings
    per day depending on how many days the dataset contains
    ----------
    metadata : full input dataset of a day, week, month, etc.

    features : columns from the metadata dataframe to be used in the distance calculation
        example ["Temperature"] or ["Temperature", "Humidity"]

    num_frames : how many frames in total do we need

    reserved_at_ends : how many points at both ends of each clip to be reserved and not sampled

    debug : for debugging purposes it also calculates and extracts addional information

    Return
    ------
    pandas dataframe containing the sampled frame metadata
    """

    # get the unique days in the dataframe
    unique_days = metadata["Folder name"].unique()
    #  number of days
    num_days = len(unique_days)

    # calculate how many samplings there need to be to get exactly the needed num_frames
    daily_frame_nums = [num_frames // num_days + (1 if x < num_frames % num_days else 0)  for x in range (num_days)]


    list_of_days = []

    counter = 0

    all_daily_smaller = []
    all_indices = []
    #  do for each day
    for day in unique_days:

        # get sub-dataframe for the current day
        curr_day = metadata[metadata["Folder name"] == day]

        # get the required number of samplings for the current day
        if debug:
            curr_day_frames,curr_smaller, curr_indices = extract_furthest_frames_metadata_daily(curr_day, features, daily_frame_nums[counter], reserved_at_ends, debug)
        else:
            curr_day_frames = extract_furthest_frames_metadata_daily(curr_day, features, daily_frame_nums[counter], reserved_at_ends, debug)
        print(f"Processing day {day}")
        # append to a list of frames
        list_of_days.append(curr_day_frames)

        if debug:
            all_daily_smaller.append(curr_smaller)
            all_indices.append(pd.Series(curr_indices))
        counter+=1

    #  transform the list of dataframes into a dataframe
    output_dataframe = pd.concat(list_of_days)

    if debug:
        return output_dataframe, all_daily_smaller, all_indices
    else:
        return output_dataframe

# debug function for visualization of the additional information that can be extracted from the extract_frames_metadata function
# the Debug variable should be set to True to get the needed daily_smaller and all_indices lists for visualization
def debug_visualization(all_daily_smaller, all_indices, max_temp, min_temp):

    
    denormalizer = (max_temp-min_temp)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for i in range(0, len(all_daily_smaller)):

        curr_day_small = all_daily_smaller[i]
        curr_day_indices = all_indices[i]



        ax.scatter(curr_day_small['Numbering']+i, (curr_day_small['Temperature']*denormalizer)+min_temp, s=5, c="blue", alpha=0.5)

        ax.scatter(curr_day_small.iloc[curr_day_indices]['Numbering']+i, (curr_day_small.iloc[curr_day_indices]['Temperature']*denormalizer)+min_temp,marker = "o", s=15, c="red", alpha=1)


    plt.xlabel('Day')
    plt.ylabel('Temperature')
    ax.set_aspect('auto')


def debug_visualization_v2(metadata, selected_metadata):
    metadata['NewTime'] = metadata['DateTime']+ pd.to_timedelta(metadata['Image Number'], unit='s')

    selected_metadata['NewTime'] = selected_metadata['DateTime']+ pd.to_timedelta(selected_metadata['Image Number'], unit='s')
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(metadata['NewTime'], metadata['Temperature'], s=5, c="blue", alpha=0.5)
    ax.scatter(selected_metadata['NewTime'], selected_metadata['Temperature'], s=5, c="red", alpha=0.5)



if __name__ == '__main__':

    metadata_path = "metadata_images.csv"

    metadata = pd.read_csv(metadata_path)

    metadata['DateTime'] = pd.to_datetime(metadata['DateTime'], dayfirst = True)

    output_dir = "splits"

    #num_frames = 100 for test # 100 or 5000 for train

    # Coldest day, week, month
    cold_day, cold_week, cold_month = get_experiment_data_v2(metadata,"min")

    cold_day_frames = extract_frames_metadata(cold_day, ["Temperature"], num_frames = 5000)
    cold_week_frames = extract_frames_metadata(cold_week, ["Temperature"], num_frames = 5000)
    cold_month_frames = extract_frames_metadata(cold_month, ["Temperature"], num_frames = 5000)
    # debug_visualization(week_small, week_inds)

    cold_day_frames.to_csv(os.path.join(output_dir,"feb_day.csv"),index=False)
    cold_week_frames.to_csv(os.path.join(output_dir,"feb_week.csv"),index=False)
    cold_month_frames.to_csv(os.path.join(output_dir,"feb_month.csv"),index=False)

    #Best case month set as January
    best_case_month = metadata[metadata["DateTime"].dt.month == 1]
    best_case_frames = extract_frames_metadata(best_case_month, ["Temperature"], num_frames = 100)
    best_case_frames.to_csv(os.path.join(output_dir,"jan_month.csv"),index=False)

    # # Hottest day, week, month
    hot_day, hot_week, hot_month = get_experiment_data_v2(metadata,"max")

    # hot_day_frames = extract_frames_metadata(hot_day, ["Temperature"], num_frames = num_frames)
    # hot_week_frames = extract_frames_metadata(hot_week, ["Temperature"], num_frames = num_frames)
    hot_month_frames = extract_frames_metadata(hot_month, ["Temperature"], num_frames = 100)

    # hot_day_frames.to_csv(os.path.join(output_dir,"hottest_day.csv"),index=False)
    # hot_week_frames.to_csv(os.path.join(output_dir,"hottest_week.csv"),index=False)
    hot_month_frames.to_csv(os.path.join(output_dir,"aug_month.csv"),index=False)

    # Median day, week, month
    mid_day, mid_week, mid_month = get_experiment_data_v2(metadata,"median")

    # mid_day_frames = extract_frames_metadata(mid_day, ["Temperature"], num_frames = num_frames)
    # mid_week_frames = extract_frames_metadata(mid_week, ["Temperature"], num_frames = num_frames)
    mid_month_frames = extract_frames_metadata(mid_month, ["Temperature"], num_frames = 100)

    # mid_day_frames.to_csv(os.path.join(output_dir,"median_day.csv"),index=False)
    # mid_week_frames.to_csv(os.path.join(output_dir,"median_week.csv"),index=False)
    mid_month_frames.to_csv(os.path.join(output_dir,"apr_month.csv"),index=False)
