# -*- coding: utf-8 -*-
"""
Created on Fri Jul 23 09:58:56 2021

@author: IvanTower
"""

import numpy as np
import pandas as pd
import os
from pathlib import Path
from scipy.spatial.distance import pdist, squareform
import copy
from scipy import stats

def distcorr(Xval, Yval, pval=True, nruns=500):
    """ Compute the distance correlation function, returning the p-value.
    Work made by wladston/distcorr.py (gist c931b1495184fbb99bec)
    Based on Satra/distcorr.py (gist aa3d19a12b74e9ab7941)
    >>> a = [1,2,3,4,5]
    >>> b = np.array([1,2,9,4,4])
    >>> distcorr(a, b)
    (0.76267624241686671, 0.266)
    """
    X = np.atleast_1d(Xval)
    Y = np.atleast_1d(Yval)
    if np.prod(X.shape) == len(X):
        X = X[:, None]
    if np.prod(Y.shape) == len(Y):
        Y = Y[:, None]
    X = np.atleast_2d(X)
    Y = np.atleast_2d(Y)
    n = X.shape[0]
    if Y.shape[0] != X.shape[0]:
        raise ValueError('Number of samples must match')
    a = squareform(pdist(X))
    b = squareform(pdist(Y))
    A = a - a.mean(axis=0)[None, :] - a.mean(axis=1)[:, None] + a.mean()
    B = b - b.mean(axis=0)[None, :] - b.mean(axis=1)[:, None] + b.mean()

    dcov2_xy = (A * B).sum() / float(n * n)
    dcov2_xx = (A * A).sum() / float(n * n)
    dcov2_yy = (B * B).sum() / float(n * n)
    dcor = np.sqrt(dcov2_xy) / np.sqrt(np.sqrt(dcov2_xx) * np.sqrt(dcov2_yy))

    if pval:
        greater = 0
        for i in range(nruns):
            Y_r = copy.copy(Yval)
            np.random.shuffle(Y_r)
            if distcorr(Xval, Y_r, pval=False) > dcor:
                greater += 1
        return (dcor, greater / float(nruns))
    else:
        return dcor


def calculate_correlations(model_metric = "MSE", correlation_type = "Pearsons", save = True):
    """ Calculate the correlations between the model metric and the other features
    Both the R value of the correlation is calculated, as well as the p-value of the statistical significance of the correlation
    The model_metric can be either "MSE" for the autoencoders and anomaly detectors or "f1_score" for the object detectors,
    The correlation_type can be either "Pearsons" or "Distance". The Distance correlation is calculated much slower especially if the p-value is needed
    The number of runs for the Distance correlation is set to 5000 and can be lower for faster computation or made higher for better p-value computation
    
    
    """
    
    if model_metric == "MSE":
        augmented_results = 'augmented_mse.csv'
    else:
        augmented_results = 'augmented_f1_score.csv'
        
    main_path = Path(os.path.dirname(__file__)).parent
    metadata_path = r"visualize_results"
    metadata = pd.read_csv(os.path.join(main_path, metadata_path, augmented_results))
    
    metadata['DateTime'] = pd.to_datetime(metadata['DateTime'], dayfirst = True)
    
    cleanup_nums = {"Day_Night":     {"day": 0, "night": 1}}
    metadata = metadata.replace(cleanup_nums)
    
    tested_columns = [model_metric,"Temperature", "Humidity", "Wind Speed", "Wind Direction", "Precipitation","Activity", "Day_Night", "Hour"] 
    
    all_correlations = []
    all_p_values = []
    for model in metadata["model"].unique():
    
        metadata_smaller = metadata[(metadata["model"]== model)].copy()
        metadata_smaller = metadata_smaller[tested_columns]
        metadata_smaller = metadata_smaller.reset_index(drop=True)
        
        print(f"For Model: {model}")
        corr_curr_model = []
        p_value_curr_model =[]
        for column in metadata_smaller.columns:
            
            if correlation_type == "Pearsons":
                cor,p = stats.pearsonr(metadata_smaller[column], metadata_smaller[model_metric])
            else:
                cor,p = distcorr(metadata_smaller[column], metadata_smaller[model_metric],nruns = 5000)
                
            corr_curr_model.append(cor)
            p_value_curr_model.append(p)
            
            print(f"between {model_metric} and {column} = R: {np.abs(cor)}, p-val: {p}")
            
        corr_curr_model.insert(0,model)
        corr_curr_model.insert(0,model_metric)
        corr_curr_model.insert(0,correlation_type)
        
        p_value_curr_model.insert(0,model)
        p_value_curr_model.insert(0,model_metric)
        p_value_curr_model.insert(0,correlation_type)
    
        print("---------------")
        all_correlations.append(corr_curr_model)
        all_p_values.append(p_value_curr_model)
        
    all_correlations_df = pd.DataFrame(all_correlations,
           columns = ["Correlation", "Metric", "Model"] + tested_columns)
    
    all_p_values_df = pd.DataFrame(all_p_values,
           columns = ["Correlation", "Metric", "Model"] + tested_columns)
        
    if save:
        all_correlations_df.to_csv(os.path.join("analysis_results", f"correlation_R_{correlation_type}_{model_metric}.csv"), index=False)
        all_p_values_df.to_csv(os.path.join("analysis_results", f"correlation_Pvalue_{correlation_type}_{model_metric}.csv"), index=False)
    
    return all_correlations_df


if __name__ == "__main__":
    
    
    metrics = ["MSE", "f1_score"]
    correlation = ["Pearsons", "Distance"]
    
    correlation_output =calculate_correlations(metrics[0], correlation[0])
    