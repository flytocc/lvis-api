import argparse
import os
import subprocess
import sys


def _find_free_port():
    import socket

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # Binding to port 0 will cause the OS to find an available port for us
    sock.bind(("", 0))
    port = sock.getsockname()[1]
    sock.close()
    # NOTE: there is still a chance the port could be taken by other processes.
    return port


def parse_args():
    """
    Helper function parsing the command line options
    @retval ArgumentParser
    """
    parser = argparse.ArgumentParser(description="PyTorch distributed testing launch "
                                                 "helper utilty that will spawn up "
                                                 "multiple distributed processes")
    # Optional arguments for the launch helper
    parser.add_argument("--gpu_id",
                        type=str,
                        default="0,1,2,3,4,5,6,7",
                        help="gpu id for test")

    parser.add_argument('--ann', type=str, default='/home/user/Database/MSCOCO2017/annotations/lvis/lvis_v1_val.json')
    parser.add_argument('--res_dir', type=str, default='/home/nieyang/Pet-dev/ckpts/cnn/LVIS/swin/centernet2-mask_SWIN-L-FPN-GCE-64ROI-MASKNORM_fed_rfs_0.5x_ms-pretrained@64ROI/res')
    parser.add_argument('--scales', type=str, default='600 700 800 900 1000 1100 1200')
    parser.add_argument('--weigths', type=str, default='45.6 47.3 48.2 48.5 48.9 48.4 48.6')
    parser.add_argument('--conf_type', choices=('avg', 'max', 'box_and_model_avg', 'absent_model_aware_avg'), type=str, default='avg')
    parser.add_argument('--overlap_thresh', type=float, default=0.55)

    return parser.parse_args()


def main():
    args = parse_args()

    # world size in terms of number of processes
    gpus = list(map(int, args.gpu_id.split(",")))
    num_gpus = len(args.gpu_id.split(","))
    dist_world_size = num_gpus

    # set PyTorch distributed related environmental variables
    current_env = os.environ.copy()
    current_env["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    current_env["MASTER_ADDR"] = "127.0.0.1"
    current_env["MASTER_PORT"] = str(_find_free_port())
    current_env["WORLD_SIZE"] = str(dist_world_size)

    processes = []

    if "OMP_NUM_THREADS" not in os.environ and num_gpus > 1:
        current_env["OMP_NUM_THREADS"] = str(1)
        # print("*****************************************\n"
        #       "Setting OMP_NUM_THREADS environment variable for each process "
        #       "to be {} in default, to avoid your system being overloaded, "
        #       "please further tune the variable for optimal performance in "
        #       "your application as needed. \n"
        #       "*****************************************".format(current_env["OMP_NUM_THREADS"]))

    for local_rank in range(num_gpus):
        # each process's rank
        dist_rank = local_rank
        current_env["RANK"] = str(dist_rank)
        current_env["LOCAL_RANK"] = str(local_rank)

        # spawn the processes
        cmd = [
            sys.executable,
            "-u", "wbf_bbox_ms_res.py",
            f"--gpu_id={gpus[local_rank]}",
            f"--local_rank={local_rank}",
            f"--ann={args.ann}",
            f"--res_dir={args.res_dir}",
            f"--scales={args.scales}",
            f"--weigths={args.weigths}",
            f"--conf_type={args.conf_type}",
            f"--overlap_thresh={args.overlap_thresh}",
        ]

        process = subprocess.Popen(cmd, env=current_env)
        processes.append(process)

    for process in processes:
        process.wait()
        if process.returncode != 0:
            raise subprocess.CalledProcessError(returncode=process.returncode, cmd=process.args)


if __name__ == "__main__":
    main()
