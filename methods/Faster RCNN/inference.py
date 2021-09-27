
import torchvision
import torch
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision import datasets, transforms
from dataloader import HarborDatasetInference
import transforms as T
from engine import train_one_epoch, evaluate
import utils
import numpy as np
import cv2

def showbbox(model, img, save=False, save_path = "inference_output/", show_delay=500):
    # The img entered is a tensor in the 0-1 range        
    model.eval()
    with torch.no_grad():
        '''
        prediction Like:
        [{'boxes': tensor([[1492.6672,  238.4670, 1765.5385,  315.0320],
        [ 887.1390,  256.8106, 1154.6687,  330.2953]], device='cuda:0'), 
        'labels': tensor([1, 1], device='cuda:0'), 
        'scores': tensor([1.0000, 1.0000], device='cuda:0')}]
        '''
        prediction = model([img.to(device)])
        
    print(prediction)
        
    img = img.permute(1,2,0)  # C,H,W_H,W,C, for drawing
    img = (img * 255).byte().data.cpu()  # * 255, float to 0-255
    img = np.array(img)  # tensor → ndarray
    
    for i in range(prediction[0]['boxes'].cpu().shape[0]):
        xmin = round(prediction[0]['boxes'][i][0].item())
        ymin = round(prediction[0]['boxes'][i][1].item())
        xmax = round(prediction[0]['boxes'][i][2].item())
        ymax = round(prediction[0]['boxes'][i][3].item())
        
        label = prediction[0]['labels'][i].item()
        
        if label == 1:
            img = cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (255, 0, 0), thickness=2)
            img = cv2.putText(img, 'Person', (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0),
                               thickness=2)

    if show_delay >= 1:
        cv2.imshow("Predictions", img)
        cv2.waitKey(show_delay)
    if save:
        cv2.imwrite(save_path, img)

def get_model(num_classes):
    # Backbone
    backbone = torchvision.models.mobilenet_v2(pretrained=False).features
    backbone.out_channels = 1280
    # Anchor generator
    anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),), aspect_ratios=((0.5, 1.0, 2.0),))
    # Featuremap names
    roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'], output_size=7, sampling_ratio=2)
    # put the pieces together inside a FasterRCNN model
    model = FasterRCNN(backbone, num_classes=num_classes, rpn_anchor_generator=anchor_generator, box_roi_pool=roi_pooler)
    return model

if __name__ == '__main__':
    batch_size = 4
    num_epochs = 100
    pretrained_weights = 'weights/pretrain.pth'
    input_ims = 'data/harborfront/validation/images/'
    save = True
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    model = get_model(num_classes = 2) #background + person
    model.to(device)
    model.load_state_dict(torch.load(pretrained_weights))

    dataset = HarborDatasetInference(input_ims, transforms=T.get_transform(train=False))

    for i in range(len(dataset)):
        img, _ = dataset[i] 
        showbbox(model, img, save, save_path="./inference_output/img{0:04d}.jpg".format(i))