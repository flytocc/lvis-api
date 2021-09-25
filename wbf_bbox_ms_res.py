import argparse
import itertools
import json
import os
from collections import defaultdict

import numpy as np
import tqdm
import torch

from ensemble_boxes_wbf import weighted_boxes_fusion

from lvis import LVIS

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
    scales = args.scales.split(' ')
    model_weights = list(map(float, args.weigths.split(' ')))

    assert len(scales) == len(model_weights)
    model_weights = np.array(model_weights)
    model_weights /= model_weights.sum()
    model_weights = model_weights.tolist()

    bbox_dict = defaultdict(list)
    for root, dirs, files in os.walk(args.res_dir):
        scale = root.split('/')[-1]
        if scale not in scales:
            continue
        for name in files:
            prefix = f'{args.prefix}{args.gpu_id}_'
            if name.startswith(prefix):
                file, ext = os.path.splitext(name)
                sub = file.split('_')[-1]
                bbox_dict[sub].append((scale, os.path.join(root, name)))

    lvis_gt = LVIS(args.ann)

    for sub_name, bbox_fn_list in bbox_dict.items():
        SAVE_PATH = os.path.join(args.res_dir, f"wbf_{args.prefix}{args.gpu_id}_{sub_name}.json")
        if os.path.exists(SAVE_PATH):
            continue

        idx = 0
        bbox_list = []
        for scale, fn in bbox_fn_list:
            idx += 1
            print(f"[{idx}/{len(bbox_fn_list)}] loading {fn}")
            with open(fn, 'r') as f:
                sub_res = json.load(f)
            for ann in sub_res:
                ann['scale'] = scale
            bbox_list.append(sub_res)
        bbox = list(itertools.chain.from_iterable(bbox_list))

        img_ids = set()
        img_bbox_map = defaultdict(lambda: defaultdict(list))
        img_scores_map = defaultdict(lambda: defaultdict(list))
        img_labels_map = defaultdict(lambda: defaultdict(list))
        print("img map")
        for ann in tqdm.tqdm(bbox):
            image_id = ann["image_id"]
            image_info = lvis_gt.load_imgs([image_id])[0]
            im_w, im_h = image_info['width'], image_info['height']
            scale = ann["scale"]
            img_ids.add(image_id)
            x1, y1, w, h = ann['bbox']
            x1 /= im_w
            y1 /= im_h
            w /= im_w
            h /= im_h
            img_bbox_map[image_id][scale].append([x1, y1, x1 + w, y1 + h])  # xyxy, [0, 1]
            img_scores_map[image_id][scale].append(ann['score'])
            img_labels_map[image_id][scale].append(ann['category_id'])

        nms_res = []
        print("wbf")
        for image_id in tqdm.tqdm(list(img_ids)):
            image_info = lvis_gt.load_imgs([image_id])[0]
            im_w, im_h = image_info['width'], image_info['height']

            bboxes_list = img_bbox_map[image_id]
            scores_list = img_scores_map[image_id]
            labels_list = img_labels_map[image_id]

            boxes, scores, labels = weighted_boxes_fusion(
                [bboxes_list[s] for s in scales],
                [scores_list[s] for s in scales],
                [labels_list[s] for s in scales],
                weights=model_weights,
                iou_thr=args.overlap_thresh,
                conf_type=args.conf_type,
            )

            boxes[:, [0, 2]] *= im_w
            boxes[:, [1, 3]] *= im_h
            boxes[:, 2:] -= boxes[:, :2]  # to xywh
            boxes = boxes.tolist()
            scores = scores.tolist()
            labels = labels.tolist()

            for b, s, l in zip(boxes, scores, labels):
                res = {
                    'image_id': image_id,
                    'category_id': l,
                    'bbox': b,
                    'score': s,
                }
                nms_res.append(res)

        # save
        print(f"save to {SAVE_PATH}")
        with open(SAVE_PATH, 'w') as f:
            json.dump(nms_res, f)

        del bbox_list
        del bbox
        del nms_res


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(description="merge_bbox_ms_res")
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--gpu_id", type=int, default=0)

    parser.add_argument('--ann', type=str, default='/home/user/Database/MSCOCO2017/annotations/lvis/lvis_v1_val.json')
    parser.add_argument('--res_dir', type=str, default='/home/nieyang/Pet-dev/ckpts/cnn/LVIS/swin/centernet2-mask_SWIN-L-FPN-GCE-64ROI-MASKNORM_fed_rfs_0.5x_ms-pretrained@64ROI/res')
    parser.add_argument('--scales', type=str, default='600 700 800 900 1000 1100 1200')
    parser.add_argument('--weigths', type=str, default='45.6 47.3 48.2 48.5 48.9 48.4 48.6')
    parser.add_argument('--conf_type', choices=('avg', 'max', 'box_and_model_avg', 'absent_model_aware_avg'), type=str, default='avg')
    parser.add_argument('--overlap_thresh', type=float, default=0.55)
    parser.add_argument('--prefix', type=str, default='bbox_')

    args = parser.parse_args()

    torch.cuda.set_device(args.local_rank)
    try:
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
    except:
        pass

    main(args)
