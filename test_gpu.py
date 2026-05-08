import torch
import time
size=10000
cpu_a=torch.randn(size,size)
cpu_b=torch.randn(size,size)
start=time.time()
cpu_res=torch.matmul(cpu_a,cpu_b)
print(f"CPU time: {time.time()-start:.2f} seconds")
device=torch.device("cuda")
gpu_a=cpu_a.to(device)
gpu_b=cpu_b.to(device)
start=time.time()
gpu_res=torch.matmul(gpu_a,gpu_b)
print(f"GPU time: {time.time()-start:.2f} seconds")