import torch

torch._dynamo.config.disable = True

import os
import numpy as np
from PIL import Image, ImageDraw
from rfdetr import RFDETRSegPreview, RFDETRMedium

model = RFDETRSegPreview()

model.train(
    dataset_dir="/home/ubuntu/work_dir/rf-detr/aug",
    epochs=30,
    batch_size=2,
    grad_accum_steps=1,
    lr=1e-4,
    output_dir="./run",
    # resume="/home/ubuntu/work_dir/rf-detr/run/checkpoint.pth",
)
