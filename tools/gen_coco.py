import argparse
import os
import json
import numpy as np
from PIL import Image
from pycocotools import mask as mask_utils
import shutil
import cv2

# ==========================
# CONFIG
# ==========================
IMG_DIR = "/home/ubuntu/work_dir/rf-detr/aug/data"
MASK_DIR = "/home/ubuntu/work_dir/rf-detr/aug/mask"

TRAIN_DIR = "/home/ubuntu/work_dir/rf-detr/aug/train"
VALID_DIR = "/home/ubuntu/work_dir/rf-detr/aug/valid"
TEST_DIR = "/home/ubuntu/work_dir/rf-detr/aug/test"

os.makedirs(TRAIN_DIR, exist_ok=True)
os.makedirs(VALID_DIR, exist_ok=True)
os.makedirs(TEST_DIR, exist_ok=True)

import torch
import pycocotools.mask as coco_mask


def convert_coco_poly_to_mask(segmentations, height, width):
    """Convert polygon segmentation to a binary mask tensor of shape [N, H, W].
    Requires pycocotools.
    """
    masks = []
    for polygons in segmentations:
        if polygons is None or len(polygons) == 0:
            # empty segmentation for this instance
            masks.append(torch.zeros((height, width), dtype=torch.uint8))
            continue
        try:
            rles = coco_mask.frPyObjects(polygons, height, width)
        except:
            rles = polygons
        mask = coco_mask.decode(rles)
        if mask.ndim < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
    if len(masks) == 0:
        return torch.zeros((0, height, width), dtype=torch.uint8)
    return torch.stack(masks, dim=0)


USE_POLYGON = True  # True: polygon, False: RLE

# ==========================
# Khởi tạo COCO dict
# ==========================
coco = {
    "info": {
        "year": "2025",
        "version": "1",
        "description": "Exported from roboflow.com",
        "contributor": "",
        "url": "https://public.roboflow.com/object-detection/undefined",
        "date_created": "2025-11-28T10:57:31+00:00",
    },
    "licenses": [
        {
            "id": 1,
            "url": "https://creativecommons.org/licenses/by/4.0/",
            "name": "CC BY 4.0",
        }
    ],
    "images": [],
    "annotations": [],
    "categories": [],  # sửa category nếu cần
}

parser = argparse.ArgumentParser()
parser.add_argument(
    "--split",
    type=str,
    required=True,
    choices=["train", "valid", "test"],
    help="Choose which split to generate",
)
args = parser.parse_args()
SPLIT = args.split

NUM_FILES = len(os.listdir(IMG_DIR))

ann_id = 1
for img_id, img_name in enumerate(sorted(os.listdir(IMG_DIR)), 1):

    ratio = img_id / NUM_FILES

    if SPLIT == "train" and ratio <= 0.8:
        dst_img_dir = TRAIN_DIR
        OUT_JSON = f"{TRAIN_DIR}/_annotations.coco.json"

    elif SPLIT == "valid" and 0.8 < ratio <= 0.9:
        dst_img_dir = VALID_DIR
        OUT_JSON = f"{VALID_DIR}/_annotations.coco.json"

    elif SPLIT == "test" and ratio > 0.9:
        dst_img_dir = TEST_DIR
        OUT_JSON = f"{TEST_DIR}/_annotations.coco.json"

    else:
        continue
    img_path = os.path.join(IMG_DIR, img_name)
    if ".jpg" in img_name:
        mask_path = os.path.join(MASK_DIR, img_name.replace(".jpg", "_mask.png"))
    elif ".png" in img_name:
        mask_path = os.path.join(MASK_DIR, img_name.replace(".png", "_mask.png"))

    shutil.copy(img_path, dst_img_dir)
    shutil.copy(mask_path, dst_img_dir)

    print(img_path, mask_path)
    # load image và mask
    image = Image.open(img_path)
    w, h = image.size
    mask = np.array(Image.open(mask_path))

    # Thêm image info
    coco["images"].append(
        {"id": img_id, "file_name": img_name, "width": w, "height": h}
    )

    coco["categories"].append({"id": img_id, "name": "vessel", "supercategory": "none"})

    # Lấy các instance id trong mask
    instance_ids = np.unique(mask)
    instance_ids = instance_ids[instance_ids != 0]  # bỏ background

    for inst_id in instance_ids:
        inst_mask = (mask == inst_id).astype(np.uint8)
        bbox = [
            int(np.min(np.where(inst_mask)[1])),
            int(np.min(np.where(inst_mask)[0])),
            int(np.max(np.where(inst_mask)[1]))
            - int(np.min(np.where(inst_mask)[1]))
            + 1,
            int(np.max(np.where(inst_mask)[0]))
            - int(np.min(np.where(inst_mask)[0]))
            + 1,
        ]

        if USE_POLYGON:
            # dùng OpenCV tìm contour -> polygon
            contours, _ = cv2.findContours(
                inst_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            polygons = [cnt.flatten().tolist() for cnt in contours if len(cnt) >= 3]
            segmentation = polygons
            iscrowd = 0
            mask_tensor = convert_coco_poly_to_mask([polygons], h, w)
        else:
            # encode mask sang RLE
            rle = mask_utils.encode(np.asfortranarray(inst_mask))
            rle["counts"] = rle["counts"].decode("utf-8")  # convert bytes → str
            segmentation = rle
            iscrowd = 1

        area = int(np.sum(inst_mask))
        if area > 0:
            coco["annotations"].append(
                {
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": 1,
                    "segmentation": segmentation,
                    "area": int(np.sum(inst_mask)),
                    "bbox": bbox,
                    "iscrowd": iscrowd,
                }
            )
            ann_id += 1


# Lưu file JSON
with open(OUT_JSON, "w") as f:
    json.dump(coco, f)
