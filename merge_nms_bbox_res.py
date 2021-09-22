import itertools
import os
import json
import warnings
from collections import defaultdict

import tqdm


def limit_dets_per_cat(anns, max_dets):
    by_cat = defaultdict(list)
    for ann in anns:
        by_cat[ann["category_id"]].append(ann)
    results = []
    fewer_dets_cats = set()
    for cat, cat_anns in tqdm.tqdm(by_cat.items()):
        if len(cat_anns) < max_dets:
            fewer_dets_cats.add(cat)
        elif len(cat_anns) > max_dets:
            cat_anns = sorted(cat_anns, key=lambda x: x["score"], reverse=True)[:max_dets]
        results.extend(cat_anns)
    if fewer_dets_cats:
        warnings.warn(
            f"{len(fewer_dets_cats)} categories had less than {max_dets} "
            f"detections!\n"
            f"Outputting {max_dets} detections for each category will improve AP "
            f"further."
        )
    return results


if __name__ == '__main__':
    RESULT_DIR = "/home/nieyang/Pet-dev/ckpts/cnn/LVIS/swin/centernet2-mask_SWIN-L-FPN-GCE_fed_rfs_1x_ms-pretrained@obj365v1/res"
    # RESULT_DIR = "../Pet-dev/ckpts/cnn/LVIS/swin/centernet2-mask_SWIN-T-FPN-GCE_fed_rfs_1x_ms/res"
    SAVE_PATH = os.path.join(RESULT_DIR, f'merged_bbox_10k_dets_per_cat.json')
    prefix = f'nms_bbox_'

    print("load bbox res")
    bbox_list = []
    for root, dirs, files in os.walk(RESULT_DIR):
        idx = 0
        for name in files:
            if name.startswith(prefix):
                idx += 1
                fn = os.path.join(root, name)
                print(f"{idx}) loading {fn}")
                with open(fn, 'r') as f:
                    sub_res = json.load(f)
                bbox_list.append(sub_res)
    bbox = itertools.chain.from_iterable(bbox_list)

    print("run limit_dets_per_cat")
    bbox_10k_dets_per_cat = limit_dets_per_cat(bbox, 10000)

    print("set ann id")
    ann_id = 1
    for ann in bbox_10k_dets_per_cat:
        ann['id'] = ann_id
        ann_id = ann_id + 1

    if len(bbox_10k_dets_per_cat) > 0:
        print(f"save to {SAVE_PATH}")
        with open(SAVE_PATH, 'w') as f:
            json.dump(bbox_10k_dets_per_cat, f)
