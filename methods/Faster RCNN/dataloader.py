import torch
import os
import numpy as np
import cv2
from torchvision import datasets, transforms
from PIL import Image
from xml.dom.minidom import parse
import glob


class HarborDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms=None, return_names=False, remove_empty=False):
        self.root = root
        self.transforms = transforms
        self.return_names =  return_names
        # load all image files, sorting them to ensure that they are aligned
        self.bbox_xml = list(sorted(os.listdir(os.path.join(root, "outputs"))))
        if remove_empty:
            self.bbox_xml = [x for x in self.bbox_xml if x[0:5] != 'empty']
        self.imgs = list([os.path.splitext(x)[0] + '.jpg' for x in self.bbox_xml])
        self.classes = {
            'background': 0,
            'person': 1,
        }
        
 
    def __getitem__(self, idx):
        # load images and bbox
        img_path = os.path.join(self.root, "images", self.imgs[idx])
        bbox_xml_path = os.path.join(self.root, "outputs", self.bbox_xml[idx])
        img = Image.open(img_path).convert("RGB")        
        
        # Read file, VOC format dataset label is xml format file
        dom = parse(bbox_xml_path)
        # Get Document Element Object
        data = dom.documentElement
        # Get objects
        objects = data.getElementsByTagName('object')        
        # get bounding box coordinates
        boxes = []
        labels = []
        for object_ in objects:
            # Get the contents of the label
            name = object_.getElementsByTagName('name')[0].childNodes[0].nodeValue  # Is label, mark_type_1 or mark_type_2
            labels.append(np.int(self.classes[name]))  # Background label is 0, mark_type_1 and mark_type_2 labels are 1 and 2, respectively
            
            bndbox = object_.getElementsByTagName('bndbox')[0]
            xmin = np.float(bndbox.getElementsByTagName('xmin')[0].childNodes[0].nodeValue)
            ymin = np.float(bndbox.getElementsByTagName('ymin')[0].childNodes[0].nodeValue)
            xmax = np.float(bndbox.getElementsByTagName('xmax')[0].childNodes[0].nodeValue)
            ymax = np.float(bndbox.getElementsByTagName('ymax')[0].childNodes[0].nodeValue)
            boxes.append([xmin, ymin, xmax, ymax])        
 
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # there is only one class
        labels = torch.as_tensor(labels, dtype=torch.int64)        
 
        image_id = torch.tensor([idx])

        if boxes.shape[0] >= 1:
            area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        else:
            area = torch.as_tensor([], dtype=torch.float32)

        # suppose all instances are not crowd
        iscrowd = torch.zeros((len(objects),), dtype=torch.int64)
 
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        # Since you are training a target detection network, there is no target [masks] = masks in the tutorial
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd
 
        if self.transforms is not None:
            img, target = self.transforms(img, target)

        if self.return_names:
            return img, target, self.imgs[idx] 
        else:
            return img, target
 
    def __len__(self):
        return len(self.imgs)

class HarborDatasetInference(torch.utils.data.Dataset):
    def __init__(self, root, transforms=None):
        self.root = root
        self.transforms = transforms
        # load all image files, sorting them to ensure that they are aligned
        self.imgs = glob.glob(root+'/*.jpg')
        self.classes = {
            'background': 0,
            'person': 1,
        }
 
    def __getitem__(self, idx):
        # load images and bbox
        img_path = self.imgs[idx]
        img = Image.open(img_path).convert("RGB")

        if self.transforms is not None:
            # Note that target (including bbox) is also transformed\enhanced here, which is different from transforms from torchvision import
            # Https://github.com/pytorch/vision/tree/master/references/detectionOfTransforms.pyThere are examples of target transformations when RandomHorizontalFlip
            img, target = self.transforms(img, [[]])

        return img, target
 
    def __len__(self):
        return len(self.imgs)