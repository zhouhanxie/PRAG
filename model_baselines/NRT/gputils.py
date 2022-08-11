import os
import torch
from tqdm import tqdm
import time


def check_mem(cuda_device):
    devices_info = os.popen('"/usr/bin/nvidia-smi" --query-gpu=memory.total,memory.used --format=csv,nounits,noheader').read().strip().split("\n")
    total, used = devices_info[int(cuda_device)].split(',')
    return total,used

def occupy_mem(cuda_device, amount='auto'):
    if amount == 'auto':
        total, used = check_mem(cuda_device)
        total = int(total)
        used = int(used)
        max_mem = int(total * 0.8)
        block_mem = max_mem - used  
    else:
        block_mem = amount

    print('[gputils]: reserving memry: ', block_mem, 'on', cuda_device)
    print('cuda availabiliy: ', torch.cuda.is_available(), ', # device', torch.cuda.device_count())
    x = torch.FloatTensor(256,1024,block_mem).cuda(device=int(cuda_device))
    del x
    print('reserved')
    
if __name__ == '__main__':
    import launcher
    cuda_device = str(launcher.load_gpuid())
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_device
    occupy_mem(cuda_device, 'auto')
    # for _ in tqdm(range(60)):
    #     time.sleep(1)
    # print('Done')