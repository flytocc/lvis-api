import argparse
import itertools
import os
import json
from collections import defaultdict

import numpy as np
import tqdm
import torch
import pycocotools.mask as mask_util

import sys
sys.path.insert(0, "/home/nieyang/Pet-dev/")
from pet.cnn.modeling.roi_head.mask.inference import _do_paste_mask


"""
segm
{
    'image_id': 9,
    'category_id': 61,
    'segmentation': rle,
    'score': 0.2566065192222595.
    'id': 1,
    'mask_prob': torch.Tensor,
    'bbox': torch.Tensor,
    'width': int,
    'height': int
}
"""

PIXEL_SCORE_TH = 0.25
QUALITY_WEIGHTS = (1.0, 0.0, 1.0)


def get_mask_results(probs, boxes, im_w, im_h):
    """
    Args:
        probs (Tensor)
        boxes (ImageContainer)

    Returns:
        rles (list[string])
        mask_pixel_scores (Tensor)
    """
    BYTES_PER_FLOAT = 4
    # TODO: This memory limit may be too much or too little. It would be better to
    # determine it based on available resources.
    GPU_MEM_LIMIT = 1024 ** 3  # 1 GB memory limit

    device = probs.device
    N, _, H, W = probs.shape

    num_chunks = N if device.type == "cpu" else int(np.ceil(N * int(im_h * im_w) * BYTES_PER_FLOAT / GPU_MEM_LIMIT))
    assert num_chunks <= N, "Default GPU_MEM_LIMIT in is too small; try increasing it"

    chunks = torch.chunk(torch.arange(N, device=device), num_chunks)
    im_masks = torch.zeros(N, im_h, im_w, device=device, dtype=torch.bool)
    im_masks_tl = torch.zeros(N, im_h, im_w, device=device, dtype=torch.bool)
    im_masks_th = torch.zeros(N, im_h, im_w, device=device, dtype=torch.bool)

    for i in chunks:
        masks_chunk, spatial_inds = _do_paste_mask(probs[i], boxes[i], im_h, im_w, skip_empty=device.type == "cpu")
        im_masks[(i,) + spatial_inds] = (masks_chunk >= 0.5).to(dtype=torch.bool)
        im_masks_tl[(i,) + spatial_inds] = (masks_chunk >= PIXEL_SCORE_TH).to(dtype=torch.bool)
        im_masks_th[(i,) + spatial_inds] = (masks_chunk >= (1 - PIXEL_SCORE_TH)).to(dtype=torch.bool)

    mask_pixel_scores = (torch.sum(im_masks_th, dim=(1, 2)).to(dtype=torch.float32)
                         / torch.sum(im_masks_tl, dim=(1, 2)).to(dtype=torch.float32).clamp(min=1e-6))

    rles = []
    for i in range(N):
        # Too slow.
        # Get RLE encoding used by the COCO evaluation API
        rle = mask_util.encode(np.array(im_masks[i].unsqueeze(2).cpu(), dtype=np.uint8, order='F'))[0]
        # For dumping to json, need to decode the byte string.
        # https://github.com/cocodataset/cocoapi/issues/70
        rle['counts'] = rle['counts'].decode('ascii')
        rles.append(rle)

    return rles, mask_pixel_scores


def main(args):
    RESULT_DIR = "../Pet-dev/ckpts/cnn/LVIS/swin/centernet2-mask_SWIN-T-FPN-GCE_fed_rfs_1x_ms/res"
    # scales = ['500', '600', '700', '800', '900', '1000', '1100', '1200']
    scales = ['700', '800']

    seg_dict = defaultdict(list)
    for root, dirs, files in os.walk(RESULT_DIR):
        scale = root.split('/')[-1]
        if scale not in scales:
            continue
        for name in files:
            prefix = f'segm_{args.gpu_id}_'
            if name.startswith(prefix):
                file, ext = os.path.splitext(name)
                sub = file.split('_')[-1]
                seg_dict[sub].append(os.path.join(root, name))

    count = 0
    for sub_name, segm_fn_list in seg_dict.items():
        SAVE_PATH = os.path.join(RESULT_DIR, f"comb_segm_{args.gpu_id}_{sub_name}.json")
        if os.path.exists(SAVE_PATH):
            continue

        count += 1
        print('-' * 50 + str(count) + '-' * 50)
        idx = 0
        segm_list = []
        for fn in segm_fn_list:
            idx += 1
            print(f"[{idx}/{len(segm_fn_list)}] loading {fn}")
            sub_res = torch.load(fn)
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

            ann = ann_list[0]
            boxes = ann['bbox']        # 4
            mask_bbox_scores = ann['bbox_score']
            image_id = ann['image_id']
            category_id = ann['category_id']
            im_w = ann['width']
            im_h = ann['height']
            masks_list = [ann['mask_prob']]   # 1, 64, 64

            for ann in ann_list[1:]:
                assert boxes.allclose(ann['bbox'])
                assert mask_bbox_scores.allclose(ann['bbox_score'])
                assert image_id == ann['image_id']
                assert category_id == ann['category_id']
                assert im_w == ann['width']
                assert im_h == ann['height']

                masks_list.append(ann['mask_prob'])

            masks = torch.stack(masks_list, dim=0)   # N, 1, 64, 64
            masks = torch.mean(masks, dim=0)

            masks = masks.unsqueeze(0)
            boxes = boxes.unsqueeze(0)

            rles, mask_pixel_scores = get_mask_results(masks, boxes, im_w, im_h)

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
    args = parser.parse_args()

    torch.cuda.set_device(args.local_rank)
    try:
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
    except:
        pass

    main(args)
