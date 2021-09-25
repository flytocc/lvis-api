import numpy as np
import torch
import pycocotools.mask as mask_util

import sys
sys.path.insert(0, "/home/nieyang/Pet-dev/")
from pet.cnn.modeling.roi_head.mask.inference import _do_paste_mask

BYTES_PER_FLOAT = 4
# TODO: This memory limit may be too much or too little. It would be better to
# determine it based on available resources.
GPU_MEM_LIMIT = 1024 ** 3  # 1 GB memory limit


def get_mask_results(probs, boxes, im_w, im_h, pixil_score_th=0.25):
    """
    Args:
        probs (Tensor)
        boxes (ImageContainer)

    Returns:
        rles (list[string])
        mask_pixel_scores (Tensor)
    """
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
        im_masks_tl[(i,) + spatial_inds] = (masks_chunk >= pixil_score_th).to(dtype=torch.bool)
        im_masks_th[(i,) + spatial_inds] = (masks_chunk >= (1 - pixil_score_th)).to(dtype=torch.bool)

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
