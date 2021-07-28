# Seasonal-Changes-in-Thermal-Surveillance-Imaging
Repo containing the code for extracting data from the "Danish Seasonal Thermal Surveillance Dataset"

[2021/XX/XX: Dataset published at X]

## Requirements

- OpenCV
- Python 3

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

