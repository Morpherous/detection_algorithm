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


def detec_test_pic(pth, pic_test):
    opt.load_path = opt.caffe_pretrain_path
    opt.env = 'detec-tset-pic'
    faster_rcnn = FasterRCNNVGG16()
    trainer = FasterRCNNTrainer(faster_rcnn).cuda()
    trainer.load(pth)
    opt.caffe_pretrain = True  # this model was trained from caffe-pretrained model
    pic_index = 0

    for pic in tqdm(os.listdir(pic_test)):
        time.sleep(1)
        img = read_image(os.path.join(pic_test, pic))
        img = t.from_numpy(img)[None]
        _bboxes, _labels, _scores = trainer.faster_rcnn.predict(img, visualize=True)
        pred_img = visdom_bbox(at.tonumpy(img[0]),
                               at.tonumpy(_bboxes[0]),
                               at.tonumpy(_labels[0]).reshape(-1),
                               at.tonumpy(_scores[0]).reshape(-1))
        trainer.vis.img('pred_img', pred_img)
        pic_index += 1
        if pic_index > 1000:
            break


detec_test_pic('checkpoints/fasterrcnn_04151330.pth_0.9007891813622001',
               '/home/data/pic/')
