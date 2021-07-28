# -*- coding: utf-8 -*-
"""
Created on Fri Jul 23 11:04:11 2021

@author: IvanTower
"""

import numpy as np
import pandas as pd

from pathlib import Path

import os

from statsmodels.tsa.stattools import adfuller, grangercausalitytests


def adf_test(df):
    """ Augmented Dicky-Fuller test
    Test if the time series is stationary or not. If the p-value is below 0.05 we consider it stationary
    """
    result = adfuller(df.values)
    print('ADF Statistics: %f' % result[0])
    print('p-value: %f' % result[1])
    print('Critical values:')
    for key, value in result[4].items():
        print('\t%s: %.3f' % (key, value))
        
        


def grangers_causation_matrix(data, variables, test='ssr_chi2test', verbose=False):    
    """Check Granger Causality of all possible combinations of the Time series.
    The rows are the response variable, columns are predictors. The values in the table 
    are the P-Values. P-Values lesser than the significance level (0.05), implies 
    the Null Hypothesis that the coefficients of the corresponding past values is 
    zero, that is, the X does not cause Y can be rejected.

    data      : pandas dataframe containing the time series variables
    variables : list containing names of the time series variables.
    """
    
    maxlag=12
    df = pd.DataFrame(np.zeros((len(variables), len(variables))), columns=variables, index=variables)
    for c in df.columns:
        for r in df.index:
            test_result = grangercausalitytests(data[[r, c]], maxlag=maxlag, verbose=False)
            p_values = [round(test_result[i+1][0][test][1],4) for i in range(maxlag)]
            if verbose: print(f'Y = {r}, X = {c}, P Values = {p_values}')
            min_p_value = np.min(p_values)
            df.loc[r, c] = min_p_value
    df.columns = [var for var in variables]
    df.index = [var + '_y' for var in variables]
    return df


def calculate_linear_granger(model_metric = "MSE", save = True):
    """ Calculate the linear Granger Causality for the tested metric 
    The model_metric can be either "MSE" for the autoencoders and anomaly detectors or "f1_score" for the object detectors
    First all the tested columns are run through an augmented Dicky-Fuller test, which yields that not all of them are stationary
    A first order differenciation is done for all columns and they are again run through the test, which this time all are stationary
    Finally the linear Granger Causality test is performed with only the causality relationship between the tested metric and the columns is saved
    """
    
    if model_metric == "MSE":
        augmented_results = 'augmented_mse.csv'
    else:
        augmented_results = 'augmented_f1_score.csv'
        
    main_path = Path(os.path.dirname(__file__)).parent
    metadata_path = r"visualize_results"
    df = pd.read_csv(os.path.join(main_path, metadata_path, augmented_results))
    
    
    tested_columns = ["Temperature", "Humidity", "Activity", "Day_Night", model_metric]

    cleanup_nums = {"Day_Night":     {"day": 0, "night": 1}}
    
    df = df.replace(cleanup_nums)
    
    
    for model in df["model"].unique():
        
    
        df_small = df[df["model"]== model].copy()
        
        df_small = df_small[tested_columns]
        
        
        
        print(f"------------ MODEL {model} ----------------")
        for column in tested_columns:
        
            print(f'ADF Test: {column} time series')
            adf_test(df_small[column])

        
    print("TRANSFORM DATA -------------")   
    
    for model in df["model"].unique():
        
        df_small = df[df["model"]== model].copy()
        
        df_small = df_small[tested_columns]
        
        df_small_transformed = df_small.diff().dropna()
        
        print(f"------------ MODEL {model} ----------------")
        for column in tested_columns:
        
            print(f'ADF Test: {column} time series')
            adf_test(df_small_transformed[column])
        
    print("LINEAR GRANGER CAUSALITY TEST ---------------")  
    
    all_models = []
    for model in df["model"].unique():
        
    
        df_small = df[df["model"]== model].copy()
        
        df_small = df_small[tested_columns]

        
        df_small_transformed = df_small.diff().dropna()
        
        print(f"------------ MODEL {model} ----------------")
        granger_df = grangers_causation_matrix(df_small_transformed, variables = df_small_transformed.columns)  
        
        output_statistic = granger_df.loc[[f"{model_metric}_y"], granger_df.columns != model_metric].squeeze().sort_values(ascending=True).to_frame().T
        
        output_statistic = output_statistic.rename(index={f"{model_metric}_y": model})
        print(output_statistic)
        
        all_models.append(output_statistic)
        
    all_models_df = pd.concat(all_models)
    
    if save:
        all_models_df.to_csv(os.path.join("analysis_results", f"linear_granger_{model_metric}.csv"))
    
    return all_models_df


if __name__ == '__main__':
    
    metrics = ["MSE", "f1_score"]
    
    granger_results = calculate_linear_granger(metrics[0])