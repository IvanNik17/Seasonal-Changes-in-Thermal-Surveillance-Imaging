# -*- coding: utf-8 -*-
"""
Created on Fri Jun  4 13:40:29 2021

@author: IvanTower
"""
import numpy as np
import pandas as pd

import os
from fnmatch import fnmatch
import re
import sys



def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]


def count_people_annotations(metadata):
    

    
    annotations_main_folder = os.path.join(os.path.dirname(os.path.realpath(__file__)), "people_manual_annotations")
    
    all_subdirs= os.listdir(annotations_main_folder)
    
    pattern = "*.txt"

    
    all_annotation_files = []
    for curr_exp in all_subdirs:
        
        
        print(f"----------------Starting {curr_exp}")
        for path, subdirs, files in os.walk( os.path.join(".", annotations_main_folder, curr_exp), topdown=True):

            subdirs.sort(key=natural_keys)
            


            for name in files:

                if fnmatch(name, pattern):   
                    
                    # all_annotation_files.append([name, curr_exp])
                    separate_parts = name.split("_")
                    
                    curr_annotation_num = len(np.loadtxt(os.path.join(path,name), delimiter=" "))
                    
                    all_annotation_files.append([int(separate_parts[0]), "_".join(separate_parts[1:4]), int(separate_parts[-1].split(".")[0]),curr_annotation_num])

    
        
    all_annotation_df = pd.DataFrame(all_annotation_files, columns=["Folder name", "Clip Name", "Image Number", "Num Annotations"])
    
    metadata_num_annot = metadata.merge(all_annotation_df, on=["Folder name", "Clip Name", "Image Number"], how='left')
    metadata_num_annot = metadata_num_annot.fillna(0)
    
    return metadata_num_annot



if __name__ == '__main__':
    
    
    metadata_path = r"D:\2021_workstuff\Seasonal-Changes-in-Thermal-Surveillance-Imaging\visualize_results\CAE\feb_month"
    metadata_name = r"results_apr.csv"
    
    
    metadata_path = os.path.join(metadata_path,metadata_name)
    metadata = pd.read_csv(metadata_path)
    metadata["DateTime"] = pd.to_datetime(metadata['DateTime'], dayfirst = True)
    
    metadata_anot = count_people_annotations(metadata)
    
    
    
    
    
    
    # annotations_main_folder = r"people_manual_annotations"
    

    # all_subdirs= os.listdir(annotations_main_folder)
    
    # pattern = "*.txt"

    
    
    # all_annotation_files = []
    # for curr_exp in all_subdirs:
        
            
    #     all_losses = []
        
        
    #     print(f"----------------Starting {curr_exp}")
    #     for path, subdirs, files in os.walk( os.path.join(".", annotations_main_folder, curr_exp), topdown=True):

    #         subdirs.sort(key=natural_keys)
            


    #         for name in files:

    #             if fnmatch(name, pattern):   
                    
    #                 # all_annotation_files.append([name, curr_exp])
    #                 separate_parts = name.split("_")
                    
    #                 curr_annotation_num = len(np.loadtxt(os.path.join(path,name), delimiter=" "))
                    
    #                 all_annotation_files.append([int(separate_parts[0]), "_".join(separate_parts[1:4]), int(separate_parts[-1].split(".")[0]),curr_annotation_num])

    
        
    # all_annotation_df = pd.DataFrame(all_annotation_files, columns=["Folder name", "Clip Name", "Image Number", "Num Annotations"])
    
    # metadata_num_annot = metadata.merge(all_annotation_df, on=["Folder name", "Clip Name", "Image Number"], how='left')
    # metadata_num_annot = metadata_num_annot.fillna(0)
    
        
        
        
        