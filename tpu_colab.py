try:
  import os 
  os.system("pip install cloud-tpu-client==0.10 torch==1.13.0 https://storage.googleapis.com/tpu-pytorch/wheels/colab/torch_xla-1.13-cp38-cp38-linux_x86_64.whl")
  os.system("pip install torch_tb_profiler")
  !pip install torch
  import torch_xla.distributed.parallel_loader as pl
  import torch_xla
  import torch_xla.core.xla_model as xm
  device = xm.xla_device()
except:
  print("TPU not found,Using Cuda ")
  device="cuda"
