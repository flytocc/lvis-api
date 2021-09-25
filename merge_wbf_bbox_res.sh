#!/usr/bin/env bash

RES_DIR="/home/nieyang/Pet-dev/ckpts/cnn/LVIS/swin/centernet2-mask_SWIN-L-FPN-GCE-64ROI-MASKNORM_fed_rfs_0.5x_ms-pretrained@64ROI/res"

PYTHON=${PYTHON:-"python"}
for SACLE in ${SCALES[@]};do
    $PYTHON merge_bbox_res.py --res_dir $RES_DIR --scale res --prefix wbf_bbox_
done
