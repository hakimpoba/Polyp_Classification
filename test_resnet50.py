import torch
import torch.nn as nn
from torchvision import models

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

weights = models.ResNet50_Weights.DEFAULT

model = models.resnet50(weights=weights)
model.fc = nn.Linear(model.fc.in_features, 2)
model = model.to(device)

# Test with one dummy image
dummy_image = torch.randn(1, 3, 224, 224).to(device)

model.eval()

with torch.no_grad():
    output = model(dummy_image)

print("Device:", device)
print("Output shape:", output.shape)
print("Model loaded successfully!")