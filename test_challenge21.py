import argparse
import time

from lvis import LVISEval


def main(args):
    if args.mode in ('bbox', 'segm'):
        lvis_eval = LVISEval(args.ann, args.res, iou_type=args.mode)
    elif args.mode == 'boundary':
        lvis_eval = LVISEval(args.ann, args.res, mode='challenge2021')
    else:
        raise ValueError(f"Unsupported mode: {args.mode}")
    lvis_eval.run()
    lvis_eval.print_results()


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(description="merge_scale_bbox_res")
    parser.add_argument('--res', type=str, default='/home/nieyang/Pet-dev/ckpts/cnn/LVIS/swin/centernet2-mask_SWIN-L-FPN-GCE-64ROI-MASKNORM_fed_rfs_0.5x_ms-pretrained@64ROI/res/merged_bbox_top13000_dets_per_cat_800.json')
    parser.add_argument('--ann', type=str, default='/home/user/Database/MSCOCO2017/annotations/lvis/lvis_v1_val.json')
    parser.add_argument('--mode', choices=('bbox', 'segm', 'boundary'), type=str, default='bbox')
    args = parser.parse_args()

    start_time = time.time()
    main(args)
    print(f"Total time: {time.time() - start_time}")
