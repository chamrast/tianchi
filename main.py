import os
import math
import time
from tqdm import tqdm
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as D
import torchvision
from torchvision import transforms as T
import albumentations as A
from args import *
from data import *


if not os.path.exists(args.log_path):
    os.makedirs(args.log_path)
train_mask = pd.read_csv(os.path.join(args.data_path, 'train_mask.csv'), sep='\t', names=['name', 'mask'])
train_mask['name'] = train_mask['name'].apply(lambda x: os.path.join(args.data_path, 'train', x))

# img = cv2.imread(train_mask['name'].iloc[0])
# mask = rle_decode(train_mask['mask'].iloc[0])
# print(rle_encode(mask) == train_mask['mask'].iloc[0])

dataset = TianChiDataset(
    train_mask['name'].values,
    train_mask['mask'].fillna('').values,
    train_trfm, False
)

_indices = np.random.permutation(len(dataset))
_n_val = len(dataset) // 7
train_idx = _indices[_n_val:]
valid_idx = _indices[:_n_val]
train_ds = D.Subset(dataset, train_idx)
valid_ds = D.Subset(dataset, valid_idx)

# define training and validation data loaders
train_loader = D.DataLoader(
    train_ds, batch_size=args.batch_size, shuffle=True, num_workers=1)

val_loader = D.DataLoader(
    valid_ds, batch_size=args.batch_size, shuffle=False, num_workers=1)


def get_model():
    model = torchvision.models.segmentation.fcn_resnet50(True)
    
#     pth = torch.load("../input/pretrain-coco-weights-pytorch/fcn_resnet50_coco-1167a1af.pth")
#     for key in ["aux_classifier.0.weight", "aux_classifier.1.weight", "aux_classifier.1.bias", "aux_classifier.1.running_mean", "aux_classifier.1.running_var", "aux_classifier.1.num_batches_tracked", "aux_classifier.4.weight", "aux_classifier.4.bias"]:
#         del pth[key]
    
    model.classifier[4] = nn.Conv2d(512, 1, kernel_size=(1, 1), stride=(1, 1))
    return model


@torch.no_grad()
def validation(model, loader, loss_fn):
    losses = []
    model.eval()
    for image, target in loader:
        image, target = image.cuda(), target.float().cuda()
        output = model(image)['out']
        loss = loss_fn(output, target)
        losses.append(loss.item())
        
    return np.array(losses).mean()


model = get_model().cuda()
optimizer = torch.optim.AdamW(
    model.parameters(), lr=1e-4, weight_decay=1e-3)


class SoftDiceLoss(nn.Module):
    def __init__(self, smooth=1., dims=(-2,-1)):

        super(SoftDiceLoss, self).__init__()
        self.smooth = smooth
        self.dims = dims
    
    def forward(self, x, y):
        tp = (x * y).sum(self.dims)
        fp = (x * (1 - y)).sum(self.dims)
        fn = ((1 - x) * y).sum(self.dims)

        dc = (2 * tp + self.smooth) / (2 * tp + fp + fn + self.smooth)
        dc = dc.mean()
        return 1 - dc


bce_fn = nn.BCEWithLogitsLoss()
dice_fn = SoftDiceLoss()


def loss_fn(y_pred, y_true):
    bce = bce_fn(y_pred, y_true)
    dice = dice_fn(y_pred.sigmoid(), y_true)
    return 0.8*bce+ 0.2*dice



if __name__ == '__main__':
    header = r'''
            Train | Valid
    Epoch |  Loss |  Loss | Time, m
    '''
    #          Epoch         metrics            time
    raw_line = '{:6d}' + '\u2502{:7.3f}' * 2 + '\u2502{:6.2f}'
    print(header)

    best_loss = math.inf
    for epoch in range(args.epoch):
        losses = []
        start_time = time.time()
        model.train()
        for image, target in tqdm(train_loader):
            image, target = image.cuda(), target.float().cuda()
            optimizer.zero_grad()
            output = model(image)['out']
            loss = loss_fn(output, target)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            # print(loss.item())
            break

        vloss = validation(model, val_loader, loss_fn)
        print(raw_line.format(epoch, np.array(losses).mean(), vloss,
                              (time.time() - start_time) / 60 ** 1))
        losses = []
        if vloss < best_loss:
            best_loss = vloss
            torch.save(model.state_dict(), os.path.join(args.log_path, 'model_best.pth'))

    print("--- test ---")
    subm = []
    model.load_state_dict(torch.load(os.path.join(args.log_path, 'model_best.pth')))
    model.eval()

    test_mask = pd.read_csv(os.path.join(args.data_path, 'test_a_samplesubmit.csv'), sep='\t', names=['name', 'mask'])
    test_mask['name'] = test_mask['name'].apply(lambda x: os.path.join(args.data_path, 'test_a', x))

    for idx, name in enumerate(tqdm(test_mask['name'].iloc[:])):
        image = cv2.imread(name)
        image = test_trfm(image)
        with torch.no_grad():
            image = image.cuda()[None]
            score = model(image)['out'][0][0]
            score_sigmoid = score.sigmoid().cpu().numpy()
            score_sigmoid = (score_sigmoid > 0.5).astype(np.uint8)
            score_sigmoid = cv2.resize(score_sigmoid, (512, 512))

            # break
        subm.append([name.split('/')[-1], rle_encode(score_sigmoid)])

    subm = pd.DataFrame(subm)
    subm.to_csv(os.path.join(args.log_path, 'tmp.csv'), index=None, header=None, sep='\t')



