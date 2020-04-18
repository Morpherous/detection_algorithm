import os
import torch as t
from utils.config import opt
from model import FasterRCNNVGG16
from trainer import FasterRCNNTrainer
from data.util import read_image
from data import VOC_BBOX_LABEL_NAMES
from utils.eval_tool import eval_detection_voc
from utils import array_tool as at

import warnings

from utils.vis_tool import visdom_bbox
from torch.utils import data as data_
import time
from data.dataset import Dataset, TestDataset, inverse_normalize
from tqdm import tqdm

warnings.filterwarnings('ignore')


def eval_set(dataloader, faster_rcnn, test_num=1000):
    pred_bboxes, pred_labels, pred_scores = list(), list(), list()
    gt_bboxes, gt_labels, gt_difficults = list(), list(), list()
    for ii, (imgs, sizes, gt_bboxes_, gt_labels_, gt_difficults_) in tqdm(enumerate(dataloader)):
        sizes = [sizes[0][0].item(), sizes[1][0].item()]
        pred_bboxes_, pred_labels_, pred_scores_ = faster_rcnn.predict(imgs, [sizes])
        gt_bboxes += list(gt_bboxes_.numpy())
        gt_labels += list(gt_labels_.numpy())
        gt_difficults += list(gt_difficults_.numpy())
        pred_bboxes += pred_bboxes_
        pred_labels += pred_labels_
        pred_scores += pred_scores_
        if ii == test_num:
            break

    result = eval_detection_voc(
        pred_bboxes, pred_labels, pred_scores,
        gt_bboxes, gt_labels, gt_difficults,
        use_07_metric=True)
    return result

opt.load_path = opt.caffe_pretrain_path
faster_rcnn = FasterRCNNVGG16()
trainer = FasterRCNNTrainer(faster_rcnn).cuda()

trainer.load('checkpoints/fasterrcnn_04151330.pth_0.9007891813622001')
opt.caffe_pretrain = True  # this model was trained from caffe-pretrained model

testset = TestDataset(opt, 'val')
test_dataloader = data_.DataLoader(testset,
                                   batch_size=1,
                                   num_workers=opt.test_num_workers,
                                   shuffle=False, \
                                   pin_memory=True
                                   )

# test all precision
eval_result = eval_set(test_dataloader, faster_rcnn, opt.test_num)
print(eval_result)
