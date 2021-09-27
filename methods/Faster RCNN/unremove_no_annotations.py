import shutil
import os
from xml.dom.minidom import parse

labels_folder = './data/harborfront/test/Jan/outputs/'


labels = os.listdir(labels_folder)
for l in labels:   
    if 'empty' in l:
        shutil.move(os.path.join(labels_folder, l), os.path.join(labels_folder, l.replace('empty', '')))