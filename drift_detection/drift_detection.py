# -*- coding: utf-8 -*-
"""
Created on Fri Jul 23 15:05:01 2021

@author: IvanTower
"""
import numpy as np
import pandas as pd
from sklearn import preprocessing
import seaborn as sns
import os
import matplotlib.pyplot as plt
from pathlib import Path

import sklearn
from sklearn import svm
from sklearn import preprocessing
from sklearn.neighbors import KernelDensity, LocalOutlierFactor
from sklearn.mixture import GaussianMixture
from sklearn.ensemble import IsolationForest
from sklearn.covariance import EllipticEnvelope

def oneClassSVM_train(inputs, outliers_fraction):
    """
    Training for the oneClassSVM, with a preset outlier_fraction of 2%
    """
    clf = svm.OneClassSVM(nu=outliers_fraction,
                          kernel="rbf",
                          gamma=0.03)
    clf.fit(inputs)
    
    return clf


def isolationForest_train(inputs, outliers_fraction):
    """
    Training for the isolation Forest, with a preset outlier_fraction of 2%
    """
    rng = np.random.RandomState(123)
    
    clf = IsolationForest(contamination=outliers_fraction,
                          max_samples="auto",
                          random_state=rng,
                          n_estimators=100)
    clf.fit(inputs)
    
    return clf


def localOutlierFactor_train(inputs, outliers_fraction, n_neighbours):
    """
    Training for the localOutlierFactor, with a preset initial contamination of 2%
    """
    clf = LocalOutlierFactor(n_neighbors=n_neighbours, contamination=outliers_fraction, novelty=True)
    clf.fit(inputs)
    
    return clf

    

def isolationForest_pred(inputs, clf):
    """
    Get the prediction score and outlier rows from the isolationForest
    Outlier are specified as those smaller than a threshold above the 10% quantile
    """
    score_pred = clf.decision_function(inputs)

    scores = clf.score_samples(inputs)
    
    thresh = np.quantile(scores, 0.1)
    
    outlier_rows = np.where(scores<=thresh)

    return score_pred,outlier_rows[0]


def oneClassSVM_pred(inputs, clf):
    """
    Get the prediction score and outlier rows from the oneClassSVM
    Outlier are specified as those smaller than a threshold above the 10% quantile
    """

    score_pred = clf.decision_function(inputs)
    
    scores = clf.score_samples(inputs)
    
    thresh = np.quantile(scores, 0.1)
    
    outlier_rows = np.where(scores<=thresh)


    return score_pred,outlier_rows[0]


def localOutlierFactor_pred(inputs, clf):
    """
    Get the prediction score and outlier rows from the localOutlierFactor
    Outlier are specified as those smaller than a threshold above the lowest 7%
    """
    
    scores = clf.score_samples(inputs)
    
    thresh = np.quantile(scores, 0.07)
    
    outlier_rows = np.where(scores<=thresh)

    return outlier_rows[0]


def find_large_drift_start(train_data, test_data, check_consecutive = 7, outlier_fraction = 0.02):
    
    cleanup_nums = {"Day_Night":     {"day": 0, "night": 1}}
    test_data = test_data.replace(cleanup_nums)
    
    test_data_full = test_data.copy()
    tested_columns = ["MSE", "Temperature", "Humidity", "Day_Night"]
    
    train_data = train_data[tested_columns]
    test_data = test_data[tested_columns]
    
    scaler = preprocessing.MinMaxScaler()
    train_data_norm = scaler.fit_transform(train_data)
    test_data_norm = scaler.fit_transform(test_data)

    train_data_norm = pd.DataFrame(train_data_norm, columns = tested_columns)
    test_data_norm = pd.DataFrame(test_data_norm, columns = tested_columns)
    
    
    clf_svm = oneClassSVM_train(train_data_norm, outliers_fraction)
    clf_if = isolationForest_train(train_data_norm, outliers_fraction)

    
    all_svm = []
    all_if = []
    
    all_score_svm = []
    all_score_if=[]
    
    
    
    # For each day in the test data use the outlier detectors to detect outlier vertices
    
    for day in test_data_full["Folder name"].unique():
        curr_test = test_data_norm[test_data_full["Folder name"] == day]
        
        score_svm, outlier_rows_svm = oneClassSVM_pred(curr_test, clf_svm)
        
        score_if, outlier_rows_if = isolationForest_pred(curr_test, clf_if)
        
        all_svm.append(outlier_rows_svm)
        
        all_score_svm.append(score_svm)
        
        all_if.append(outlier_rows_if)
        
        all_score_if.append(score_if)        
        
        print(f"day {day} finished")
        
    all_problem_elements = []
    all_tested_days = test_data_full["Folder name"].unique()
    num_outliers = []
    for i in range(0, len(all_svm)):
        curr_problems = list(set.intersection(*map(set, [all_svm[i], all_if[i]])))
        all_problem_elements.append(curr_problems)
        
        num_outliers.append(len(curr_problems))
        
        print(f"for day {all_tested_days[i]} - {len(curr_problems)} ")

    
    # Combine the start of each 7 days with the counted outliers for each of these 7 day intervals 
    days_outliers_combined = np.vstack((all_tested_days[::check_consecutive], np.add.reduceat(num_outliers, np.arange(0, len(num_outliers), check_consecutive)))).T  
    
    return days_outliers_combined


if __name__ == "__main__":


    train_path = 'cae_results_metadata_February_training.csv'
    test_path_pt1 = 'cae_results_metadata_fullDataset_testing_pt1.csv'
    test_path_pt2 = 'cae_results_metadata_fullDataset_testing_pt2.csv'
    
    check_consecutive = 7
    outliers_fraction = 0.02


    train_data = pd.read_csv(train_path)
    test_data_pt1 = pd.read_csv(test_path_pt1)
    test_data_pt2 = pd.read_csv(test_path_pt2)
    
    
    test_data = pd.concat([test_data_pt1, test_data_pt2])
    
    test_data = test_data.reset_index()
    
    
    
    train_data['DateTime'] = pd.to_datetime(train_data['DateTime'], dayfirst = True)
    test_data['DateTime'] = pd.to_datetime(test_data['DateTime'], dayfirst = True)
    

    find_large_drift_start(train_data, test_data, check_consecutive = 7, outlier_fraction = 0.02)
    
    
    
    