import torch
from torch_points3d.applications.semseg import PointNet2

# Load pretrained PointNet++ model for ScanNet
model = PointNet2(pretrained=True, dataset="scannet")

# Print model summary
print(model)
