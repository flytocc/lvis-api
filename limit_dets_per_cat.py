import argparse
import itertools
import json
import os
import time
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
    # Parse arguments
    parser = argparse.ArgumentParser(description="merge_scale_bbox_res")
    parser.add_argument('--res', type=str, default='/home/nieyang/Pet-dev/ckpts/cnn/LVIS/swin/centernet2-mask_SWIN-L-FPN-GCE-64ROI-MASKNORM_fed_rfs_0.5x_ms-pretrained@64ROI/bbox_15k_final_50.6.json')
    parser.add_argument('--limit', type=int, default=12000)
    args = parser.parse_args()

    path = os.path.split(args.res)[0]
    SAVE_PATH = os.path.join(path, f"top{args.limit}.json")
    if os.path.exists(SAVE_PATH):
        exit(0)

    start_time = time.time()

    print("load res")
    with open(args.res, 'r') as f:
        results = json.load(f)

    print("run limit_dets_per_cat")
    limited_results = limit_dets_per_cat(results, args.limit)

    if len(limited_results) > 0:
        print(f"save to {SAVE_PATH}")
        with open(SAVE_PATH, 'w') as f:
            json.dump(limited_results, f)

    print(f"Total time: {time.time() - start_time}")
