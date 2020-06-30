from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from pointnet.dataset import ShapeNetDataset, ModelNetDataset
from pointnet.model import PointNetCls, feature_transform_regularizer
import torch.nn.functional as F
from tqdm import tqdm


parser = argparse.ArgumentParser()
parser.add_argument(
    '--batchSize', type=int, default=32, help='input batch size')
parser.add_argument(
    '--num_points', type=int, default=2500, help='input batch size')
parser.add_argument(
    '--model_num', type=str, default='0', required=True)
parser.add_argument(
    '--workers', type=int, help='number of data loading workers', default=4)
parser.add_argument('--dataset', default='/data1/pointnet_test/shapenetcore_partanno_segmentation_benchmark_v0', type=str, help="dataset path")
parser.add_argument('--dataset_type', type=str, default='shapenet', help="dataset type shapenet|modelnet40")
parser.add_argument('--feature_transform', action='store_true', help="use feature transform")

opt = parser.parse_args()
print(opt)

opt.manualSeed = random.randint(1, 10000)  # fix seed
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

if opt.dataset_type == 'shapenet':
    dataset = ShapeNetDataset(
        root=opt.dataset,
        classification=True,
        npoints=opt.num_points)

    test_dataset = ShapeNetDataset(
        root=opt.dataset,
        classification=True,
        split='test',
        npoints=opt.num_points,
        data_augmentation=False)
elif opt.dataset_type == 'modelnet40':
    dataset = ModelNetDataset(
        root=opt.dataset,
        npoints=opt.num_points,
        split='trainval')

    test_dataset = ModelNetDataset(
        root=opt.dataset,
        split='test',
        npoints=opt.num_points,
        data_augmentation=False)
else:
    exit('wrong dataset type')

testdataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=opt.batchSize,
        shuffle=True,
        num_workers=int(opt.workers))

print(len(dataset), len(test_dataset))
num_classes = len(dataset.classes)
print('classes', num_classes)

if opt.model_num=='all':
    print('Compute all model...')
    acc_list = []
    for i in range(250):
        print('model %d' % i)
        model_num = i

        classifier = PointNetCls(k=num_classes, feature_transform=opt.feature_transform)

        model_path = 'trained/cls/cls_model_'+ str(model_num) + '.pth'
        classifier.load_state_dict(torch.load(model_path))


        optimizer = optim.Adam(classifier.parameters(), lr=0.001, betas=(0.9, 0.999))
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
        classifier.cuda()


        total_correct = 0
        total_testset = 0
        for i,data in tqdm(enumerate(testdataloader, 0)):
            points, target = data
            target = target[:, 0]
            points = points.transpose(2, 1)
            points, target = points.cuda(), target.cuda()
            classifier = classifier.eval()
            pred, _, _ = classifier(points)
            pred_choice = pred.data.max(1)[1]
            correct = pred_choice.eq(target.data).cpu().sum()
            total_correct += correct.item()
            total_testset += points.size()[0]

        acc_list.append(total_correct / float(total_testset))

    import numpy as np
    import matplotlib.pyplot as plt
    accs = np.array(acc_list)
    best_model_num = np.argmax(accs)
    best_acc = np.max(accs)

    plt.figure()
    plt.plot(accs)
    plt.title('best model:%d / best acc:%.4f' % (best_model_num, best_acc))
    plt.grid()
    plt.show()


else:
    print('Compute only model '+opt.model_num+'...')
    model_num = int(opt.model_num)

    classifier = PointNetCls(k=num_classes, feature_transform=opt.feature_transform)

    model_path = 'trained/cls/cls_model_'+ str(model_num) + '.pth'
    classifier.load_state_dict(torch.load(model_path))

    optimizer = optim.Adam(classifier.parameters(), lr=0.001, betas=(0.9, 0.999))
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    classifier.cuda()


    total_correct = 0
    total_testset = 0
    for i,data in tqdm(enumerate(testdataloader, 0)):
        points, target = data
        target = target[:, 0]
        points = points.transpose(2, 1)
        points, target = points.cuda(), target.cuda()
        classifier = classifier.eval()
        pred, _, _ = classifier(points)
        pred_choice = pred.data.max(1)[1]
        correct = pred_choice.eq(target.data).cpu().sum()
        total_correct += correct.item()
        total_testset += points.size()[0]

    print('accuracy is {:.6f}'.format(total_correct / float(total_testset)))