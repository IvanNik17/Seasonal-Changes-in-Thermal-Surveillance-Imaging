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
from scipy.ndimage import gaussian_filter1d, uniform_filter1d

import os
import sys

from statsmodels.tsa.seasonal import seasonal_decompose

path = os.path.join(os.path.dirname(__file__), os.pardir)

if path not in sys.path:
    sys.path.append(path)


from pre_processing.augment_results import augment_dataframe



# from loaders.pytorch_lightning.dataset import Dataset



def calculate_trend(df, column = "MSE", decomp_freq = 30):
    trend_df = None
    
    for model in models:
        
        df_curr = df[df["model"] == model]
        decomposition = seasonal_decompose(df_curr[column], freq=decomp_freq) 
        trend_ = decomposition.trend
        
        if trend_df is None:
            trend_df = trend_
        else:
            trend_df = trend_df.append(trend_)
            
    return trend_df


def plot_error_vs_other(df, columns, save_dir, smooth, normalize, decomp_freq=30):
    if smooth:
        # # old smooth using statsmodels for time series data
        # smooth_column = calculate_trend(df, column = columns[1], decomp_freq = decomp_freq)
        # df["smooth"] = smooth_column

        # new smoothening (needs to be verified for testing)
        df = df.reset_index()
        old_columns1 = f"{columns[1]}_pre_smooth"
        df = df.rename(columns={columns[1]: old_columns1})
        df[columns[1]] = np.nan
        for model in df['model'].unique():
            mask = df['model'] == model
            assert sum(mask) == 300
            df_model = df[mask]
            df_model_sorted = df_model.sort_values(by=[columns[0]])
            df_model_sorted['smooth'] = uniform_filter1d(df_model_sorted[old_columns1], size=10)
            for index, row in df_model_sorted.iterrows():
                df.loc[index, columns[1]] = row['smooth']
        assert df[columns[1]].isnull().values.any() == 0

    if normalize:
        df = df.reset_index()
        old_columns1 = f"{columns[1]}_pre_norm"
        df = df.rename(columns={columns[1]: old_columns1})
        df[columns[1]] = np.nan
        for model in df['model'].unique():
            mask = df['model'] == model
            assert sum(mask) == 300
            df_model = df[mask]
            min_val =  df_model[old_columns1].min()
            max_val =  df_model[old_columns1].max()
            df_model['norm'] = (df_model[old_columns1] - min_val) / (max_val - min_val)
            min_val =  df_model['norm'].min()
            max_val =  df_model['norm'].max()
            assert min_val == 0
            assert max_val == 1
            for index, row in df_model.iterrows():
                df.loc[index, columns[1]] = row['norm']
        assert df[columns[1]].isnull().values.any() == 0

    if "DateTime" in columns:
        df["DateTime"] = pd.to_datetime(df['DateTime'], dayfirst=True).dt.strftime('%m-%d')

    f, ax = plt.subplots(figsize=(10, 5))
    pl = sns.lineplot(ax=ax, data=df, x=columns[0], y=columns[1], hue="model", ci="sd")

    if columns[0] == 'DateTime':
        pl.xaxis.set_major_locator(ticker.MultipleLocator(15))
        # plt.xticks(rotation=45)
    elif columns[0] == 'Hour':
        pl.xaxis.set_major_locator(ticker.MultipleLocator(1))
        plt.xlim(0, 24)

    save_filename = f'plot_{columns[0]}_{columns[1]}'
    if smooth:
        save_filename += '_smooth'
    if normalize:
        save_filename += '_normalize'
    save_dir = os.path.join(save_dir, save_filename + '.png')
    save_figure(f, save_dir)
    # plt.show()
    
    
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


def save_figure(figure, destination_path, save_pdf=False):
    destination_dirname = os.path.dirname(destination_path)
    if not os.path.exists(destination_dirname):
        os.makedirs(destination_dirname)

    figure_file_path = destination_path
    figure.savefig(figure_file_path, bbox_inches='tight', pad_inches=0.1, dpi=100)

    if save_pdf:
        figure_file_path_pdf = figure_file_path.replace('.png', '.pdf')
        figure.savefig(figure_file_path_pdf, bbox_inches='tight', pad_inches=0.1, dpi=100)

def cae_datetime_mse():
    method_folder = "CAE"

    models = ['feb_day', 'feb_week', 'feb_month']
    months = ['results_jan', 'results_apr', 'results_aug']

    df = None
    for model in models:
        for month in months:
            path = os.path.join(method_folder,model,month+'.csv')

            df_ = pd.read_csv(path)
            # df_["DateTime"] = pd.to_datetime(df_['DateTime'], dayfirst = True).dt.strftime('%m-%d')
            df_["DateTime"] = pd.to_datetime(df_['DateTime'], dayfirst=True)

            df_['model'] = model
            df_['month'] = month

            if df is None:
                df = df_
            else:
                df = df.append(df_)

    save_folder = os.path.join(method_folder, "results")
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    # df = augment_dataframe(df)

    # calculate_correlation_mat(df,save_folder)
    # calculate_depended_value_corr(df, save_folder)

    plot_error_vs_other(df, ["DateTime", "MSE"], save_folder, smooth=False)
    # plot_error_vs_other(df,["Temperature", "MSE"],save_folder,smooth = False)
    # plot_error_vs_other(df,["Humidity", "MSE"],save_folder,smooth = False)

    # plot_barplots(df, ["Hour", "MSE"], save_folder)
    # plot_barplots(df, ["Day_Night", "MSE"], save_folder)
    # plot_barplots(df, ["Timeslot_name", "MSE"], save_folder)
    # plot_barplots(df, ["Weekday_name", "MSE"], save_folder)


def models_plot(measurement_x, measurement_y, smooth=False, normalize=False, augment_from_file=False):
    models = ['CAE', 'VQVAE', 'MNAD_recon', 'MNAD_pred']
    splits = ['feb_month']
    months = ['results_jan', 'results_apr', 'results_aug']

    df = None
    for model in models:
        for split in splits:
            for month in months:
                path = os.path.join(model, split, month+'.csv')

                df_ = pd.read_csv(path)
                # df_["DateTime"] = pd.to_datetime(df_['DateTime'], dayfirst = True).dt.strftime('%m-%d')
                df_["DateTime"] = pd.to_datetime(df_['DateTime'], dayfirst=True)

                df_['model'] = model
                df_['split'] = split
                df_['month'] = month

                if df is None:
                    df = df_
                else:
                    df = df.append(df_)

    save_folder = os.path.join("results")
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    augment_save_path = os.path.join('augmented.csv')
    if augment_from_file:
        # augment missing columns (activity etc.)
        df_ = pd.read_csv(augment_save_path)
        assert len(df) == len(df_)
        missing_columns = list(set(df_.columns) - set(df.columns))
        for missing_column in missing_columns:
            df[missing_column] = df_[missing_column]
    else:
        df = augment_dataframe(df)
        # # save augmented dataframe for future loading
        # df.to_csv(augment_save_path, index=False)

    plot_error_vs_other(df, [measurement_x, measurement_y], save_folder, smooth, normalize)


if __name__ == '__main__':
    smooth = True
    normalize = True
    augment_from_file = True

    # # drift over time
    # cae_datetime_mse()

    # drift over environmental factors
    # models_plot('Temperature', 'MSE', smooth, normalize, augment_from_file)
    # models_plot('Humidity', 'MSE', smooth, normalize, augment_from_file)
    # no drift over environmental factors
    # models_plot('Wind Speed', 'MSE', smooth, normalize, augment_from_file)
    # models_plot('Wind Speed', 'MSE_moving_bkgrnd', smooth, normalize,  augment_from_file)
    # models_plot('SunPos_azimuth', 'MSE', smooth, normalize, augment_from_file)
    # models_plot('SunPos_zenith', 'MSE', smooth, normalize, augment_from_file)

    # # no drift over activity
    # models_plot('Hour', 'Num Annotations', smooth, normalize, augment_from_file)
    # models_plot('Hour', 'Activity', smooth, normalize, augment_from_file)
    # models_plot('Num Annotations', 'MSE', smooth, normalize, augment_from_file)
    # models_plot('Num Annotations', 'MSE_moving_bkgrnd', smooth, normalize, augment_from_file)
    # models_plot('Hour', 'MSE', smooth, normalize, augment_from_file)
    # models_plot('Hour', 'MSE_moving_bkgrnd', smooth, normalize, augment_from_file)
    # models_plot('Activity', 'MSE', smooth, normalize, augment_from_file)
    # models_plot('Activity', 'MSE_moving_bkgrnd', smooth, normalize, augment_from_file)
