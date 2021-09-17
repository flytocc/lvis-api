import itertools
import json
import os

from lvis import LVIS, LVISEval, LVISResults

LVIS.sub_size = 2005000


ANNOTATION_PATH = "/home/user/Database/COCO/annotations/lvis/lvis_v1_val.json"
RESULT_DIR = "../Pet-dev/ckpts/cnn/LVIS/swin/centernet2-mask_SWIN-T-FPN-GCE_fed_rfs_1x_ms/res"
RESULT_PATH = os.path.join(RESULT_DIR, 'merged_segm_10k_dets_per_cat.json')

print("create lvis_gt")
lvis_gt = LVIS(ANNOTATION_PATH, precompute_boundary=True)

# bbox_list = []
# for root, dirs, files in os.walk(RESULT_DIR):
#     for name in files:
#         if name.startswith('bbox_'):
#             fn = os.path.join(root, name)
#             with open(fn, 'r') as f:
#                 sub_res = json.load(f)
#             bbox_list.append(sub_res)
# bbox = list(itertools.chain.from_iterable(bbox_list))

# bbox_dt = LVISResults(lvis_gt, bbox)
# bbox_eval = LVISEval(lvis_gt, bbox_dt, iou_type='bbox')
# bbox_eval.run()
# bbox_eval.print_results()

# del bbox_eval, bbox_dt, bbox

print("load segm res")
with open(RESULT_PATH, 'r') as f:
    segm = json.load(f)

# segm_dt = LVISResults(lvis_gt, segm)
# segm_eval = LVISEval(lvis_gt, segm_dt, iou_type='segm')
# segm_eval.run()
# segm_eval.print_results()

# del segm_eval, segm_dt

print("create boundary_dt")
boundary_dt = LVISResults(
    ANNOTATION_PATH, segm,
    max_dets_per_cat=10000,
    max_dets_per_im=-1,
    precompute_boundary=True)

print("create boundary_eval")
boundary_eval = LVISEval(lvis_gt, boundary_dt, iou_type='boundary', mode='challenge2021')

print("run boundary_eval")
boundary_eval.run()
boundary_eval.print_results()
