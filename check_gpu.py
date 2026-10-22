import torch
import torchvision

print("PyTorch version:", torch.__version__)
print("Torchvision version:", torchvision.__version__)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("CUDA version used by PyTorch:", torch.version.cuda)

    x = torch.rand(3, 3).to("cuda")
    print("Tensor device:", x.device)
else:
    print("GPU is not available to PyTorch.")