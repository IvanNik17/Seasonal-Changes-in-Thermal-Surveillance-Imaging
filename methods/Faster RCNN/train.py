
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
    best = 0
    batch_size = 4
    num_epochs = 100
    pretrained_weights = 'weights/pretrain.pth'
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    model = get_model(num_classes=2) #background + person
    model.to(device)
    
    if pretrained_weights != '':
        model.load_state_dict(torch.load(pretrained_weights))

    dataset_train = HarborDataset('data/harborfront/test/Apr/', transforms=T.get_transform(train=True), remove_empty=True)
    dataset_val = HarborDataset('data/harborfront/100 from Feb 2021 + 100 from March 2021/val/', transforms=T.get_transform(train=False), remove_empty=True)

    dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=2, collate_fn=utils.collate_fn, drop_last=True)
    dataloader_val = torch.utils.data.DataLoader(dataset_val, batch_size=1, shuffle=True, num_workers=2, collate_fn=utils.collate_fn)


    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.0005, momentum=0.9, weight_decay=0.005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    for epoch in range(num_epochs):
        # train for one epoch, printing every 10 iterations
        train_one_epoch(model, optimizer, dataloader_train, device, epoch, print_freq=10)
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset

        res = evaluate(model, dataloader_val, device=device)
        if not os.path.exists('weights/best.pth'):
            # Save best model
            torch.save(model.state_dict(), 'weights/best.pth')
            best = res.coco_eval['bbox'].stats[1]
        elif best <= res.coco_eval['bbox'].stats[1]:
            # Save best model
            torch.save(model.state_dict(), 'weights/best.pth')
            best = res.coco_eval['bbox'].stats[1]
            print("Saving weights with {} AP50".format(best))