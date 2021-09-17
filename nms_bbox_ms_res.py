import argparse
import itertools
import os
import json
import warnings
from collections import defaultdict

import numpy as np
import tqdm
import torch

import sys
sys.path.insert(0, "/home/nieyang/Pet-dev/")
from pet.lib.ops import ml_nms


"""
bbox
{
    'image_id': 9,
    'category_id': 61,
    'bbox': [303.3284606933594, 1.341711401939392, 330.8099060058594, 239.7283172607422],
    'score': 0.2566065192222595
}
"""


def main(args):
    RESULT_DIR = "../Pet-dev/ckpts/cnn/LVIS/swin/centernet2-mask_SWIN-T-FPN-GCE_fed_rfs_1x_ms/res"
    SAVE_PATH = os.path.join(RESULT_DIR, f"rank_{args.local_rank}.json")
    
    overlap_thresh = 0.51
    topk = -1

    bbox_dict = defaultdict(list)
    for root, dirs, files in os.walk(RESULT_DIR):
        for name in files:
            prefix = f'bbox_{args.local_rank}_'
            if name.startswith(prefix):
                file, ext = os.path.splitext(name)
                sub = file.split('_')[-1]
                bbox_dict[sub].append(os.path.join(root, name))

    nms_res = []
    for bbox_fn_list in bbox_dict.values():
        bbox_list = []
        for fn in bbox_fn_list:
            with open(fn, 'r') as f:
                sub_res = json.load(f)
            bbox_list.append(sub_res)
        bbox = list(itertools.chain.from_iterable(bbox_list))

        img_ids = set()
        img_bbox_map = defaultdict(list)
        img_scores_map = defaultdict(list)
        img_labels_map = defaultdict(list)
        for ann in bbox:
            image_id = ann["image_id"]
            img_ids.add(image_id)
            img_bbox_map[image_id].append(ann['bbox'])
            img_scores_map[image_id].append(ann['score'])
            img_labels_map[image_id].append(ann['category_id'])

        for image_id in list(img_ids):
            bboxes_list = img_bbox_map[image_id]
            scores_list = img_scores_map[image_id]
            labels_list = img_labels_map[image_id]

            bboxes_np = np.array(bboxes_list, dtype=np.float32)
            scores_np = np.array(scores_list, dtype=np.float32)
            labels_np = np.array(labels_list, dtype=np.int64)
            bboxes = torch.from_numpy(bboxes_np).cuda()
            scores = torch.from_numpy(scores_np).cuda()
            labels = torch.from_numpy(labels_np).cuda()
            bboxes[:, 2:] += bboxes[:, :2]

            keep = ml_nms(bboxes, scores, labels, overlap_thresh, topk).cpu().tolist()

            for k in keep:
                res = {
                    'image_id': image_id,
                    'category_id': labels_list[k],
                    'bbox': bboxes_list[k],
                    'score': scores_list[k],
                }
                nms_res.append(res)

        del sub_res, bbox_list, bbox

    # save
    print(f"save to {SAVE_PATH}")
    with open(SAVE_PATH, 'w') as f:
        json.dump(nms_res, f)


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(description="merge_bbox_ms_res")
    parser.add_argument("--local_rank", type=int, default=0)
    args = parser.parse_args()

    torch.cuda.set_device(args.local_rank)
    try:
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
    except:
        pass

    main(args)
