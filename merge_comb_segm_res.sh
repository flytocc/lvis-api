#!/usr/bin/env bash

RES_DIR="/home/nieyang/Pet-dev/ckpts/cnn/LVIS/swin/centernet2-mask_SWIN-L-FPN-GCE-64ROI-MASKNORM_fed_rfs_0.5x_ms-pretrained@64ROI/res"

PYTHON=${PYTHON:-"python"}

$PYTHON merge_res.py --res_dir $RES_DIR --scale comb_segm --prefix comb_segm_ --limit 10000
