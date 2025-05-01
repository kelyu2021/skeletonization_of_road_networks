from torchviz import make_dot
import torch
from models.unet import UNet

model = UNet(1, 1)
output = model(torch.randn(1, 1, 256, 256))
make_dot(output, params=dict(model.named_parameters())).render("unet_graph", format="png")