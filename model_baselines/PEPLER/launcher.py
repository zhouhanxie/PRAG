# launch program based on gpu stats.. use gracefully

import pynvml
import os
import time
import torch
pynvml.nvmlInit()


def get_gpu_status(gpu_idx=0):
    handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_idx)
    ratio = 1024**2
    meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
    total = meminfo.total/ratio
    used = meminfo.used/ratio
    free = meminfo.free/ratio
    return total, used, free


def save_gpuid(gpuid):
    torch.save(gpuid, 'launcher_current_gpu.pt')

def load_gpuid():
    return torch.load('launcher_current_gpu.pt')

def main(instruction, mem_needed=1000):

    cmd = 'export CUDA_DEVICE_ORDER=PCI_BUS_ID &&'+instruction # 'export CUDA_VISIBLE_DEVICES=@availid@ && '+instruction
    

    while 1:
        avail = None
        for gpuid in [0,1,2,3]:
            total, used, free = get_gpu_status(gpuid)
            print(str(gpuid)+" free: ", free, end='\r')

            if free > mem_needed:
                print()
                print('gpu ', gpuid,' is free with avail mem ', free)
                print("start")
                save_gpuid(gpuid)
                os.system(cmd.replace('@availid@', str(gpuid)))
                print("finish")
                return 

            time.sleep(0.25)
            print(' '*40, end='\r')

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--command", required=False, type=str, default="echo use --command", help="command to run")
    args = parser.parse_args()
    main(
        instruction = args.command,
        mem_needed=6000
        )
    

