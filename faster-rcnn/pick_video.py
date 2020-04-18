import os
import torch as t
from utils.config import opt
from model import FasterRCNNVGG16
from trainer import FasterRCNNTrainer
from data.util import read_image

import warnings
from utils.vis_tool import vis_bbox
from utils import array_tool as at
from utils.vis_tool import visdom_bbox

warnings.filterwarnings('ignore')

from tqdm import tqdm
import time

opt.load_path = opt.caffe_pretrain_path
opt.env = 'detec-tset-pic'
faster_rcnn = FasterRCNNVGG16()
trainer = FasterRCNNTrainer(faster_rcnn).cuda()
trainer.load('checkpoints/fasterrcnn_04151330.pth_0.9007891813622001')
opt.caffe_pretrain = True  # this model was trained from caffe-pretrained model


def judge_animal(img_path):
    img = read_image(img_path)
    img = t.from_numpy(img)[None]
    _bboxes, _labels, _scores = trainer.faster_rcnn.predict(img, visualize=True)
    print(_labels)


img_path = ''
if judge_animal(img_path):
    print('Have animale')
else:
    print('Do not detect animale')
