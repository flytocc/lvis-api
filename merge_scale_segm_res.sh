#!/usr/bin/env bash

RES_DIR="/home/nieyang/Pet-dev/ckpts/cnn/LVIS/swin/centernet2-mask_SWIN-L-FPN-GCE-64ROI-MASKNORM_fed_rfs_0.5x_ms-pretrained@64ROI/res"
SCALES=$1  # "600 800 1000 1200"

PYTHON=${PYTHON:-"python"}
for SACLE in ${SCALES[@]};do
    $PYTHON merge_segm_res.py --res_dir $RES_DIR --scale $SACLE --prefix split_segm_
done
