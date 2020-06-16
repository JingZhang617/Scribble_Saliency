import torch
import torch.nn.functional as F

import numpy as np
import pdb, os, argparse
from scipy import misc

from model.vgg_models import Back_VGG
from data import test_dataset
os.environ["CUDA_VISIBLE_DEVICES"] = '1'

parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=352, help='testing size')
opt = parser.parse_args()

dataset_path = './test/img/'

model = Back_VGG(channel=32)
model.load_state_dict(torch.load('./models/scribble_30.pth'))

model.cuda()
model.eval()

test_datasets = ['ECSSD','DUT','DUTS_Test','THUR','HKU-IS']

for dataset in test_datasets:
    save_path = './results/ResNet50/' + dataset + '/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    image_root = dataset_path + dataset + '/'
    test_loader = test_dataset(image_root, opt.testsize)
    for i in range(test_loader.size):
        print i
        image, HH,WW,name = test_loader.load_data()
        image = image.cuda()
        res0,res1,res= model(image)
        res = F.upsample(res, size=[WW,HH], mode='bilinear', align_corners=False)
        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        misc.imsave(save_path+name, res)