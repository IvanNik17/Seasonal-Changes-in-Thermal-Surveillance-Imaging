# -*- coding: utf-8 -*-
"""
Created on Fri Jul 23 13:22:09 2021

@author: IvanTower
"""

import torch
import numpy as np
import pandas as pd
from sklearn import preprocessing
import seaborn as sns
import os
from pathlib import Path

import matplotlib.pyplot as plt

from models.cmlp import cMLP, cMLPSparse, train_model_ista, train_unregularized
from models.clstm import cLSTM, train_model_ista


def nonlinear_granger(model_metric = "MSE", trained_model = "mlp"):
    """ Nonlinear granger causality
    Based on the work in 
    Alex Tank, Ian Covert, Nicholas Foti, Ali Shojaie, Emily Fox. "Neural Granger Causality." Transactions on Pattern Analysis and Machine Intelligence, 2021
    
    GitHub: https://github.com/iancovert/Neural-GC
    
    The model_metric can be either "MSE" for the autoencoders and anomaly detectors or "f1_score" for the object detectors
    The trained_model can choose between MLP and LSTM model for calculating the granger causality. The model is trained using the provided hyperparameters and the its hidden layers are used for the causality calculation
    
    Before using for training the models a first order differenciation is done to the columns and they are normalized
    Both the trained granger model is saved together with the loss for each epoch and the final calculated causality matrix
    """

    # For GPU acceleration
    device = torch.device('cuda')
    
    
    # import my data
    
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
    
    
    tested_columns = [model_metric,"Temperature", "Humidity","Day_Night","Activity"]
    
    models_save = "models_saved"
    
    all_train_results = []
    for model in metadata["model"].unique():
        
        metadata_small = metadata[metadata["model"]== model].copy()
        
        metadata_small = metadata_small[tested_columns]
        
        print(model)
        
        metadata_small = metadata_small.diff().dropna()
    
        metadata_small = metadata_small.astype(np.float32)
        
        scaler = preprocessing.MinMaxScaler()
        normalized_arr = scaler.fit_transform(metadata_small)
    
        
     
        curr_features = torch.tensor(normalized_arr[np.newaxis], dtype=torch.float32, device=device)
        
        
        
        # Set up model
        if trained_model == "mlp":
            
            granger_model = cMLP(curr_features.shape[-1], lag=6, hidden=[100]).cuda(device=device)
            
            # Train with ISTA
            train_loss_list = train_model_ista(
                granger_model, curr_features, lam=0.002, lam_ridge=1e-2, lr=5e-2, penalty='H', max_iter=50000,
                check_every=100)
            
        else:
            granger_model = cLSTM(curr_features.shape[-1], hidden=100).cuda(device=device)
            
            # Train with ISTA
            train_loss_list = train_model_ista(
                granger_model, curr_features, context=10, lam=0.001, lam_ridge=1e-3, lr=5e-2, max_iter=50000,
                check_every=50)
        
        
    
    
        
        # Verify learned Granger causality
        GC_est = granger_model.GC(False).cpu().data.numpy()
        
        all_train_results.append([train_loss_list,GC_est])
    
    
        torch.save(granger_model, os.path.join(models_save, f"model_{trained_model}_detector_{model}.pt"))
        
        print(f"End model for {model}")
        
    
    
    save_dir = r"analysis_results"
        
    # visualize calculated causality and save to csv 
    for i in range(0, len (metadata["model"].unique())):    
        # Loss function plot
        fig, axarr = plt.subplots(1, 1, figsize=(8, 5))
        axarr.plot(50 * np.arange(len(all_train_results[i][0])), all_train_results[i][0])
        plt.title(f' {trained_model} training {metadata["model"].unique()[i]}')
        plt.ylabel('Loss')
        plt.xlabel('Training steps')
        plt.tight_layout()
        plt.show()
        
        
        f, ax = plt.subplots(figsize=(10, 10))
        causation_curr = all_train_results[i][1]
        causation_curr[causation_curr > 0] = 1
        
        
        heatmap = sns.heatmap(causation_curr,ax = ax, annot=True, yticklabels = tested_columns, xticklabels = tested_columns, cbar = True)
        heatmap.set_title(f"Causation Matrix {metadata['model'].unique()[i]}")
        ax.set_yticklabels(tested_columns, rotation=90, va="center") 
        plt.show()
        
        
        
        save_dir_causal_mat = os.path.join(save_dir, f'{trained_model}_causal_mat_' + str(metadata["model"].unique()[i]) + '.csv')
        np.savetxt(save_dir_causal_mat, all_train_results[i][1], delimiter=",")
        
        save_dir_loss = os.path.join(save_dir, f'{trained_model}_granger_loss_' + str(metadata["model"].unique()[i]) + '.csv')
        np.savetxt(save_dir_loss, all_train_results[i][0], delimiter=",")

    
    


if __name__ == "__main__":

    metrics = ["MSE", "f1_score"]
    granger_model = ["mlp", "lstm"]
    
    nonlinear_granger(model_metric=metrics[0],trained_model=granger_model[0] )