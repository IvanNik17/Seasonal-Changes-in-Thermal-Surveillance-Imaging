
import torchvision
import torch
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision import datasets, transforms
from dataloader import HarborDataset
import transforms as T
from engine import train_one_epoch, evaluate
import utils
import os


def get_model(num_classes):
    # Backbone
    backbone = torchvision.models.mobilenet_v2(pretrained=False).features
    backbone.out_channels = 1280
    # Anchor generator
    anchor_generator = AnchorGenerator(sizes=((20, 30, 40, 50, 60, 70, 80, 90),), aspect_ratios=((0.4405, 0.8644, 0.8120, 0.8693, 0.7680),))
    # Featuremap names
    roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'], output_size=7, sampling_ratio=2)
    # put the pieces together inside a FasterRCNN model
    model = FasterRCNN(backbone, num_classes=num_classes, rpn_anchor_generator=anchor_generator, box_roi_pool=roi_pooler)
    return model


if __name__ == '__main__':
    batch_size = 4
    pretrained_weights = 'weights/feb2021-march2021.pth'
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    model = get_model(num_classes = 2) #background + person
    model.to(device)
    model.load_state_dict(torch.load(pretrained_weights))

    dataset = HarborDataset('data/harborfront/test/Aug/', transforms=T.get_transform(train=False))

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2, collate_fn=utils.collate_fn, drop_last=False)

    # params = [p for p in model.parameters() if p.requires_grad]
 
    res = evaluate(model, dataloader, device=device)