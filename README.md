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

Outputs csv files for each data split in 'splits/'
```bash
python setup_experiments.py
```

**Load data using samples specified in csv files**

Under 'loaders/' are examples of loaders for loading images and metadata using the csv files in 'splits/'

Make sure to update paths to extracted images' folder and 'splits/'

```bash
python datamodule.py
```
