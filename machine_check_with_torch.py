
import torch

print("torch version:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name())
    print("the number of cuda devices:", torch.cuda.device_count())
    print("current cuda device:", torch.cuda.current_device())
    print("(major, minor) cuda capability", torch.cuda.get_device_capability())
