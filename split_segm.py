import argparse
import json
import os

import torch
import tqdm


def main(args):
    scales = args.scales.split(' ')

    for root, dirs, files in os.walk(args.res_dir, followlinks=True):
        scale = root.split('/')[-1]
        if scale not in scales:
            continue

        idx = 0
        for name in files:
            if name.startswith(args.prefix):
                filename, ext= os.path.splitext(name)
                if ext != '.pth':
                    continue

                SAVE_PATH = os.path.join(root, filename + '_split.json')
                if os.path.exists(SAVE_PATH):
                    continue

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
                    r.pop("bbox_score")

                print(f"save to {SAVE_PATH}")
                with open(SAVE_PATH, 'w') as f:
                    json.dump(sub_res, f)


if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser(description="split_segm")
    parser.add_argument('--res_dir', type=str, default='/home/nieyang/Pet-dev/ckpts/cnn/LVIS/swin/centernet2-mask_SWIN-L-FPN-GCE-64ROI-MASKNORM_fed_rfs_0.5x_ms-pretrained@64ROI/res')
    parser.add_argument('--scales', type=str, default='1200')
    parser.add_argument('--prefix', type=str, default='segm_')

    args = parser.parse_args()

    main(args)
