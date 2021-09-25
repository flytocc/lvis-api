import argparse
import itertools
import os
import json
from collections import defaultdict

import tqdm
import torch
import torch.nn.functional as F

from mask_utils import get_mask_results

"""
segm
{
    'image_id': 9,
    'category_id': 61,
    'segmentation': rle,
    'score': 0.2566065192222595.
    'id': 1,
    'mask_prob': torch.Tensor,
    'bbox_score': torch.Tensor,
    'bbox': torch.Tensor,
    'width': int,
    'height': int
}
"""

AREAS = [(0, 32**2), (32**2, 96**2), (96**2, float('inf'))]
FUSION_BY_SIZE = [False, False, True]   # fusion only large

PIXEL_SCORE_TH = 0.25
QUALITY_WEIGHTS = (1.0, 0.0, 1.0)

QUALITY_BASE_FUSION = True
USE_SOFTMAX = True
USE_MAX = False


def main(args):
    use_qbf = QUALITY_BASE_FUSION
    use_max = USE_MAX
    use_softmax = USE_SOFTMAX
    scales = args.scales.split(' ')
    main_scale = args.main_scale

    assert any(FUSION_BY_SIZE)
    assert not (use_max == True and use_softmax == True), "use_softmax or use_max"
    assert main_scale in scales

    seg_dict = defaultdict(list)
    for root, dirs, files in os.walk(args.res_dir, followlinks=True):
        scale = root.split('/')[-1]
        if scale not in scales:
            continue
        for name in files:
            prefix = f'{args.prefix}{args.gpu_id}_'
            if name.startswith(prefix):
                file, ext = os.path.splitext(name)
                sub = file.split('_')[-1]
                seg_dict[sub].append((scale, os.path.join(root, name)))

    count = 0
    for sub_name, segm_fn_list in seg_dict.items():
        filename = f"comb_{args.prefix}{args.gpu_id}_{sub_name}"
        if use_max:
            filename = filename + "_max"
        elif use_softmax:
            filename = filename + "_softmax"
        filename = filename + ".json"

        SAVE_PATH = os.path.join(args.res_dir, f"comb_{args.prefix}", filename)
        if os.path.exists(SAVE_PATH):
            continue

        count += 1
        print('-' * 50 + f'{count}/{len(seg_dict)}' + '-' * 50)
        idx = 0
        segm_list = []
        for scale, fn in segm_fn_list:
            idx += 1
            print(f"[{idx}/{len(segm_fn_list)}] loading {fn}")
            sub_res = torch.load(fn)
            for ann in sub_res:
                ann['scale'] = scale
            segm_list.append(sub_res)
        segm = list(itertools.chain.from_iterable(segm_list))

        ann_ids = set()
        id_ann_map = defaultdict(list)
        print("img map")
        for ann in tqdm.tqdm(segm):
            ann_id = ann["id"]
            ann_ids.add(ann_id)
            id_ann_map[ann_id].append(ann)

        comb_res = []
        print("comb")
        for ann_id in tqdm.tqdm(list(ann_ids)):
            ann_list = id_ann_map[ann_id]
            assert len(scales) == len(ann_list)

            ann = ann_list[0]
            boxes = ann['bbox']        # 4, xyxy
            mask_bbox_scores = ann['bbox_score']
            image_id = ann['image_id']
            category_id = ann['category_id']
            im_w = ann['width']
            im_h = ann['height']
            masks_list = [ann['mask_prob']]   # 1, 64, 64
            weights_list = [ann['score']]
            scales_list = [ann['scale']]

            to_fusion = False
            area = (boxes[3] - boxes[1]) * (boxes[2] - boxes[0])
            for (tl, th), flag in zip(AREAS, FUSION_BY_SIZE):
                if flag and (tl <= area < th):
                    to_fusion = True
                    break

            for ann in ann_list[1:]:
                assert boxes.allclose(ann['bbox'])
                assert mask_bbox_scores.allclose(ann['bbox_score'])
                assert image_id == ann['image_id']
                assert category_id == ann['category_id']
                assert im_w == ann['width']
                assert im_h == ann['height']

                masks_list.append(ann['mask_prob'])
                weights_list.append(ann['score'])
                scales_list.append(ann['scale'])

            if to_fusion:
                masks = torch.stack(masks_list, dim=0)   # N, 1, 64, 64

                if not use_qbf:
                    masks = torch.mean(masks, dim=0)
                else:
                    weights_list = torch.tensor(weights_list)
                    if use_softmax:
                        weights = F.softmax(weights_list, dim=0)
                    elif use_max:
                        weights = torch.zeros_like(weights_list)
                        weights[weights_list.argmax()] = 1.0
                    else:
                        weights_norm = weights_list.sum()
                        if weights_norm != 0:
                            weights = weights_list / weights_norm
                        else:
                            weights = torch.full_like(weights_list, 1.0 / len(weights_list))
                    masks *= weights.view(-1, 1, 1, 1)
                    masks = torch.sum(masks, dim=0)
            else:
                for scale, _mask in zip(scales_list, masks_list):
                    if scale == main_scale:
                        masks = _mask
                        break

            masks = masks.unsqueeze(0)
            boxes = boxes.unsqueeze(0)

            rles, mask_pixel_scores = get_mask_results(masks, boxes, im_w, im_h, pixil_score_th=PIXEL_SCORE_TH)

            mask_iou_scores = mask_bbox_scores
            alpha, beta, gamma = QUALITY_WEIGHTS
            _dot = (torch.pow(mask_bbox_scores, alpha)
                    * torch.pow(mask_iou_scores, beta)
                    * torch.pow(mask_pixel_scores, gamma))
            score = torch.pow(_dot, 1. / sum((alpha, beta, gamma))).item()

            res = {
                'image_id': image_id,
                'category_id': category_id,
                'segmentation': rles[0],
                'score': score,
            }
            comb_res.append(res)

        # save
        print(f"save to {SAVE_PATH}")
        with open(SAVE_PATH, 'w') as f:
            json.dump(comb_res, f)

        del segm_list
        del segm
        del comb_res


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(description="merge_bbox_ms_res")
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--gpu_id", type=int, default=0)

    parser.add_argument('--res_dir', type=str, default='/home/nieyang/Pet-dev/ckpts/cnn/LVIS/swin/centernet2-mask_SWIN-L-FPN-GCE-64ROI-MASKNORM_fed_rfs_0.5x_ms-pretrained@64ROI/res')
    parser.add_argument('--scales', type=str, default='600 800 1000 1200')
    parser.add_argument('--main_scale', type=str, default='1200')
    parser.add_argument('--prefix', type=str, default='segm_')

    args = parser.parse_args()

    torch.cuda.set_device(args.local_rank)
    try:
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
    except:
        pass

    main(args)
