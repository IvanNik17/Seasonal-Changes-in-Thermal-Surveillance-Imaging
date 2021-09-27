import shutil
import os
from xml.dom.minidom import parse

labels_folder = './data/harborfront/test/Apr/outputs/'


labels = os.listdir(labels_folder)
for l in labels:
    dom = parse(os.path.join(labels_folder, l))
    # Get Document Element Object
    data = dom.documentElement
    # Get objects
    objects = data.getElementsByTagName('object')        
    
    if len(objects) <=0 :
        shutil.move(os.path.join(labels_folder, l), os.path.join(labels_folder, 'empty' + l))