#%%
from __future__ import print_function
import argparse
import os
import random
import torch
from pointnet.dataset import ShapeNetDataset
from pointnet.model import PointNetDenseCls, feature_transform_regularizer
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument(
    '--model_num', type=int, default=24)
parser.add_argument(
    '--data_num', type=int, default=0)
parser.add_argument(
    '--show', type=str, default='pred', help='pred | true')
opt = parser.parse_args()

#%%
dataset_path = '/data1/pointnet_test/shapenetcore_partanno_segmentation_benchmark_v0'
class_choice = 'Chair'
model_num = opt.model_num
data_num = opt.data_num



dataset = ShapeNetDataset(
    root=dataset_path,
    classification=False,
    class_choice=[class_choice])

test_dataset = ShapeNetDataset(
    root=dataset_path,
    classification=False,
    class_choice=[class_choice],
    split='test',
    data_augmentation=False)

# print(len(dataset), len(test_dataset))
num_classes = dataset.num_seg_classes
print('segmentation classes', num_classes)

classifier = PointNetDenseCls(k=num_classes, feature_transform=False)

model_path = 'trained/seg/seg_model_'+ 'Chair_' + str(model_num) + '.pth'
classifier.load_state_dict(torch.load(model_path))

classifier.cuda()

#%%
points, target = dataset[data_num]
points.unsqueeze_(0)
target.unsqueeze_(0)
points = points.transpose(2, 1)
points, target = points.cuda(), target.cuda()
classifier = classifier.eval()
pred, _, _ = classifier(points)
pred_choice = pred.data.max(2)[1]

pred_np = pred_choice.cpu().data.numpy()
target_np = target.cpu().data.numpy() - 1
predicted = pred_np.squeeze(0)
truth = target_np.squeeze(0)


#%%


def gen_pcd(xyz, label):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    # max_label = label.max()
    colors = plt.get_cmap('tab10')(label / 10)
    pcd.colors = o3d.utility.Vector3dVector(colors[:,:3])
    return pcd

xyz = points.cpu().numpy().squeeze(0).transpose(1, 0)

if opt.show == 'pred':
    pcd_pred = gen_pcd(xyz, predicted)
    o3d.visualization.draw_geometries([pcd_pred], window_name='predict', width=800, height=600)
elif opt.show == 'true':
    pcd_true = gen_pcd(xyz, truth)
    o3d.visualization.draw_geometries([pcd_true], window_name='true', width=800, height=600)
else:
    print('show type error')

