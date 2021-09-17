import json
import os
import time

from lvis import LVIS, LVISEval, LVISResults

LVIS.sub_size = 2005000


ANNOTATION_PATH = "/home/user/Database/COCO/annotations/lvis/lvis_v1_val.json"
RESULT_DIR = "../Pet-dev/ckpts/cnn/LVIS/swin/centernet2-mask_SWIN-T-FPN-GCE_fed_rfs_1x_ms/res"
RESULT_PATH = os.path.join(RESULT_DIR, 'merged_segm_10k_dets_per_cat.json')

print("create lvis_gt")
tic = time.time()
lvis_gt = LVIS(ANNOTATION_PATH, precompute_boundary=True)
print('create lvis_gt (t={:0.2f}s)'.format(time.time() - tic))

print("load segm res")
tic = time.time()
with open(RESULT_PATH, 'r') as f:
    segm = json.load(f)
print('load segm res (t={:0.2f}s)'.format(time.time() - tic))

print("create boundary_dt")
tic = time.time()
boundary_dt = LVISResults(
    ANNOTATION_PATH, segm,
    max_dets_per_cat=10000,
    max_dets_per_im=-1,
    precompute_boundary=True)
print('create boundary_dt (t={:0.2f}s)'.format(time.time() - tic))

print("create boundary_eval")
tic = time.time()
boundary_eval = LVISEval(lvis_gt, boundary_dt, iou_type='boundary', mode='challenge2021')
print('create boundary_eval (t={:0.2f}s)'.format(time.time() - tic))

print("run boundary_eval")
tic = time.time()
boundary_eval.run()
print('run boundary_eval (t={:0.2f}s)'.format(time.time() - tic))

boundary_eval.print_results()
