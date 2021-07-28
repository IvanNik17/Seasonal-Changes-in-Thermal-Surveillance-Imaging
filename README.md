# Seasonal-Changes-in-Thermal-Surveillance-Imaging
Repo containing the code for extracting data from the "Danish Seasonal Thermal Surveillance Dataset"

[2021/XX/XX: Dataset published at X]

## Requirements

- OpenCV
- Python 3
- Pandas
- Numpy
- Pytorch
- Scikit-learn
- Seaborn (optional)
- Scipy

## Usage

**Download and extract dataset**

Download from X and extract both daily video directories in a suitable location

```bash
wget ...
```

**Extract frames from video**

Remember to change the path to the video dir

```bash
python extract_images.py
```

**Create data splits for experiments (optional)**

Outputs csv files for each data split in 'splits/'. The default number of frames that are selected per split is 100. This number can be increased to 5000 incase labels are not needed.

```bash
python setup_experiments.py
```

**Load data using samples specified in csv files**

Under 'loaders/' are examples of loaders for loading images and metadata using the csv files in 'splits/'

Make sure to update paths to extracted images' folder and 'splits/'

```bash
python datamodule.py
```

**Training provided models**

Under the 'methods/' folder each of the tested deep learning models has a separate folder. In that folder there are scripts for for running the training, testing or the full experiment. Make sure to update the paths for the images, metadata and splits in the necessary scripts

For CAE autoencoder - run both training and testing
```bash
python run_exp.py -exp month -path "/path_where_images_are_located" -train True
```

For VQVAE2 autoencoder - run both training and testing
(check the config.py if model parameters need to be changed)
```bash
python run_exp.py -exp month -path "/path_where_images_are_located" -train True
```

For MNAD_pred and MNAD_recon

For training
```bash
python Train.py -method recon -dataset_type one_month -exp_dir "/dir_to_save_trained_model" -datasplit_dir "/dir_where_dataspits_are"
```
For testing
```bash
python test_harbor.py -method recon -dataset_path "/path_where_images_are_located" -model_dir "/dir_where_trained_model_is"
```

For YOLOv5 
```bash
python ..."
```

For Faster R-CNN 
```bash
python ..."
```

**Visualizing and Augmenting Results**

For creating visualizations for the results run the visualize_results.py. The script can run from the already computed and augmented result .csvs - augmented_mse.csv (for CAE, VQVAE2, MNAD_pred, MNAD_recon) and augmented_f1_score.csv (for YOLOv5 and faster R-CNN). If you want ot run your own results through the visualization and augmentation the *augment_from_file* needs to be set top True in the visualize_results.py in the plotting functions.

For visualization of results from CAE, VQVAE2, MNAD_pred, MNAD_recon
```python
smooth = True # if smoothing the plot is required
normalize = True # if normalization of the plot between 0 and 1 is required
augment_from_file = True # if you want to use the provided already augmented results
models_plot('Temperature', 'MSE', smooth, normalize, augment_from_file)
```

For visualization of results from YOLOv5 and faster R-CNN
```python
smooth = True # if smoothing the plot is required
normalize = True # if normalization of the plot between 0 and 1 is required
augment_from_file = True # if you want to use the provided already augmented results
object_detection_plot('Temperature', 'f1_score', smooth, normalize, augment_from_file)
```

If the results only need to be augmented, without visualization the script augment_results.py in the '/pre_processing' folder can be used by first changing the necessary paths to the dataset images

```python
import pre_processing.augmented_results as aug_res
augmented = aug_res.augment_dataframe(output_csv_from_model)
```

**Correlation Analysis**

For calculating the correlation results from the MSE and f1_scores of the tested models the script 'correlation_analysis.py' from the folder '/analysis' can be used. The script searches for the augmented_mse.csv (for CAE, VQVAE2, MNAD_pred, MNAD_recon) and augmented_f1_score.csv (for YOLOv5 and faster R-CNN) files in the '/visualize_results'. All the results from the script are saved in the '/analysis_results' folder. The script takes two types of metrics - MSE and f1_score and can calculate two types of correlation - Pearsons and Distance (based on the work by https://gist.github.com/wladston/c931b1495184fbb99bec). Both correlations calculate both the 'R' and 'p-value'

```python
save = True # if the results should be saved
metrics = ["MSE", "f1_score"]
correlation = ["Pearsons", "Distance"]
correlation_output =calculate_correlations(metrics[0], correlation[0], save = save)
```

**Linear Granger Analysis**

For calculating the linear Granger causality for the MSE and f1_scores use the script 'linear_granger_causality.py' from the folder '/analysis'. The script searches for the augmented_mse.csv (for CAE, VQVAE2, MNAD_pred, MNAD_recon) and augmented_f1_score.csv (for YOLOv5 and faster R-CNN) files in the '/visualize_results'. All the results from the script are saved in the '/analysis_results' folder. The script takes two types of metrics - MSE and f1_score. It first calculates the Augmented Dicky-Fuller test over the provided data both before and after a first order differenciation. Granger causality test values need to be at least below 0.05 for a statistically significant granger causality to exist.

```python
save = True # if the results should be saved
metrics = ["MSE", "f1_score"] 
granger_results = calculate_linear_granger(metrics[0], save = save)
```

**Non-Linear Granger Analysis**

For calculating the non-linear Granger causality for the MSE and f1_scores use the script 'nonlinear_granger_causality.py' from the folder '/analysis'. The script is based on the work of https://github.com/iancovert/Neural-GC. The non-linear Granger causality is calculated using both LSTM and MLP architecture. The script searches for the augmented_mse.csv (for CAE, VQVAE2, MNAD_pred, MNAD_recon) and augmented_f1_score.csv (for YOLOv5 and faster R-CNN) files in the '/visualize_results'. All the results from the script are saved in the '/analysis_results' folder. The script takes two types of metrics - MSE and f1_score. The LSTM or MLP has to be first trained on the provided data before the causality can be calculated. For the used hyperparameters please look into the provided script. The calculated causality matrix is also visualized. Everything that is not granger causal is set to 0.

```python
metrics = ["MSE", "f1_score"] #used metric
granger_model = ["mlp", "lstm"] #used model for detecting the causality
nonlinear_granger(model_metric=metrics[0],trained_model=granger_model[0] )
```

**Detect Drift**
For testing the drift detection the script 'drift_detection.py' in the '/drift_detection' folder can be used. It uses the February training split, together with the results from the CAE model MSE. The testing data is the metadata of the full dataset together with the augmentation of the Day_Night and the results of the CAE model MSE run on the entire dataset. The test set is split into two .csvs so it meets github file size limitations. 

The drift detection uses IsolationForests and one Class SVM provided as part of the scikit-learn library. It trains the algorithms on the training data and runs them on the test data of each day, computing the sum of detected 'outliers' for each day and for each week. For the hyperparameters used in the two methods please check the provided script.

```python
check_consecutive = 7 #after calculating the number of detected outliers for each day, how many consecutive days should be summed - in this case a whole week
outliers_fraction = 0.02 # even in the training data some values can be outliers, how large the percentage of this outliers is
#....
# import the training and testing data as train_data and test_data, and set the "DateTime" column to pandas datatime format
#....
find_large_drift_start(train_data, test_data, check_consecutive = check_consecutive, outlier_fraction = outliers_fraction)
```




