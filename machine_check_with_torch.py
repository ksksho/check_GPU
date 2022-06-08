
import torch

print("torch version:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU(now):", torch.cuda.get_device_name())
    print("the number of cuda devices:", torch.cuda.device_count())
    for idx in range(torch.cuda.device_count()):
        print(f"cuda:{idx}, {torch.cuda.get_device_name(idx)}")
    print("current cuda device:", torch.cuda.current_device())
    print("(major, minor) cuda capability", torch.cuda.get_device_capability())
print("cudnn available:", torch.backends.cudnn.is_available())
if torch.backends.cudnn.is_available():
    print("cudnn version:", torch.backends.cudnn.version())
    print("cudnn:", torch.backends.cudnn.enabled)
