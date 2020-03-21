import argparse
import os
import pickle as pkl
import numpy as np
from detection.utils.list_util import load_img_list
# from pycocotools.coco import COCO
#
#
# dataset_split_mapping = {
#     "train2014": "train2014",
#     "val2014": "val2014",
#     "valminusminival2014": "val2014",
#     "minival2014": "val2014",
#     "train2017": "train2017",
#     "val2017": "val2017",
#     "test-dev2017": "test2017",
#     "train": "train"
# }


def parse_args():
    parser = argparse.ArgumentParser(description='Generate SimpleDet GroundTruth Database')
    # parser.add_argument('--dataset', help='dataset name', type=str)
    # parser.add_argument('--dataset-split', help='dataset split, e.g. train2017, minival2014', type=str)
    parser.add_argument('--img_lst_path', help='path to ds')
    parser.add_argument('--data_root', help='path to root')

    args = parser.parse_args()
    return args.img_lst_path, args.data_root


def generate_groundtruth_database(img_lst_path, data_root):

    total_rec_list = load_img_list(img_lst_path, data_root)

    # img_ids = None
    version = 1

    roidb = []
    for i, item in enumerate(total_rec_list):
        roi_rec = {
            'image_url': item['img_path'],
            'im_id': 'det1.2_{}'.format(i),
            'h': 576,
            'w': 1024,
            'gt_class': item['boxes'][:, 4], # list of class id + offset
            'gt_bbox': item['boxes'],
            'gt_poly': None,
            'version': version,
            'flipped': False}

        roidb.append(roi_rec)

    return roidb


if __name__ == "__main__":
    d, dsplit = parse_args()
    roidb = generate_groundtruth_database(d, dsplit)
    os.makedirs("data/cache", exist_ok=True)
    with open("data/cache/%s_%s.roidb" % (d, dsplit), "wb") as fout:
        pkl.dump(roidb, fout)
