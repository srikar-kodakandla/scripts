try:
  import os
  device_name = os.environ['COLAB_TPU_ADDR']
  TPU_ADDRESS = 'grpc://' + device_name
  print('Found TPU')
  print(TPU_ADDRESS)
  os.system("pip install cloud-tpu-client==0.10 torch==1.13.0 https://storage.googleapis.com/tpu-pytorch/wheels/colab/torch_xla-1.13-cp38-cp38-linux_x86_64.whl")
  os.system("pip install torch_tb_profiler")
  
  import torch_xla.distributed.parallel_loader as pl
  import torch_xla
  import torch_xla.core.xla_model as xm
  print("If you run this for the first time ,Restart the Colab Instance and run this cell again")
  device = xm.xla_device()
  device_name="tpu"
except:
  import torch
  if torch.cuda.is_available():
    print("TPU not found,GPU found,Using Cuda ")
    device="cuda"
    device_name="cuda"
  else:
    print("TPU and GPU not found,Using CPU ")
    device="cpu"
    device_name="cpu"
