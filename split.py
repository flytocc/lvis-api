import itertools
import os
import json
import warnings
from collections import defaultdict

import torch
import tqdm


if __name__ == '__main__':
    RESULT_DIR = "../Pet-dev/ckpts/cnn/LVIS/swin/centernet2-mask_SWIN-T-FPN-GCE_fed_rfs_1x_ms/res"
    scales = ['800']
    prefix = 'segm_'

    segm_list = []
    for root, dirs, files in os.walk(RESULT_DIR):
        scale = root.split('/')[-1]
        if scale not in scales:
            continue
        idx = 0
        for name in files:
            if name.startswith(prefix):
                idx += 1
                fn = os.path.join(root, name)
                print(f"{idx}) loading {fn}")
                sub_res = torch.load(fn)
                for r in sub_res:
                    r.pop("id")
                    r.pop("mask_prob")
                    r.pop("width")
                    r.pop("height")
                    r.pop("bbox")

                name, ext = os.path.splitext(fn)
                SAVE_PATH = name + '_s' + '.json'
                print(f"save to {SAVE_PATH}")
                with open(SAVE_PATH, 'w') as f:
                    json.dump(sub_res, f)
