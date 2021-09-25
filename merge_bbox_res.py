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


def main(args):
    SAVE_PATH = os.path.join(args.res_dir, f'merged_{args.prefix}top{args.limit}_dets_per_cat_{args.scale}.json')
    if os.path.exists(SAVE_PATH):
        warnings.warn(f'file existed: {SAVE_PATH}')
        return

    print("load bbox res")
    bbox_fn_list = []
    for root, dirs, files in os.walk(args.res_dir, followlinks=True):
        scale = root.split('/')[-1]
        if args.scale != scale:
            continue
        for name in files:
            if name.startswith(args.prefix):
                fn = os.path.join(root, name)
                bbox_fn_list.append(fn)

    bbox_list = []
    for idx, fn in enumerate(bbox_fn_list, 1):
        print(f"[{idx}/{len(bbox_fn_list)}] loading {fn}")
        with open(fn, 'r') as f:
            sub_res = json.load(f)
        bbox_list.append(sub_res)
    bbox = itertools.chain.from_iterable(bbox_list)

    print("run limit_dets_per_cat")
    bbox_topk_dets_per_cat = limit_dets_per_cat(bbox, args.limit)

    if args.reset_id:
        print("set ann id")
        ann_id = 1
        for ann in bbox_topk_dets_per_cat:
            ann['id'] = ann_id
            ann_id = ann_id + 1

    if len(bbox_topk_dets_per_cat) > 0:
        print(f"save to {SAVE_PATH}")
        with open(SAVE_PATH, 'w') as f:
            json.dump(bbox_topk_dets_per_cat, f)


if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser(description="merge_scale_bbox_res")
    parser.add_argument('--res_dir', type=str, default='/home/nieyang/Pet-dev/ckpts/cnn/LVIS/swin/centernet2-mask_SWIN-L-FPN-GCE-64ROI-MASKNORM_fed_rfs_0.5x_ms-pretrained@64ROI/res')
    parser.add_argument('--scale', type=str, default='800')
    parser.add_argument('--prefix', type=str, default='bbox_')
    parser.add_argument('--limit', type=int, default=12000)
    parser.add_argument('--reset_id', action='store_true')
    args = parser.parse_args()

    start_time = time.time()
    main(args)
    print(f"Total time: {time.time() - start_time}")
