import torch
from models.unet import UNet

model = UNet(1, 1)

x = torch.randn(1, 1, 256, 256)

torch.onnx.export(model, x, "unet.onnx", opset_version=11)