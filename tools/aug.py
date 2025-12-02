import os
import random
from PIL import Image, ImageOps
import numpy as np

IMG_DIR = "/home/ubuntu/work_dir/rf-detr/datasets/data"
MASK_DIR = "/home/ubuntu/work_dir/rf-detr/datasets/mask"


IMG_PATHS = sorted([os.path.join(IMG_DIR, fname) for fname in os.listdir(IMG_DIR)])
MASK_PATHS = sorted([os.path.join(MASK_DIR, fname) for fname in os.listdir(MASK_DIR)])

# ============================
# CONFIG
# ============================
OUT_IMG_DIR = "/home/ubuntu/work_dir/rf-detr/aug/data"
OUT_MASK_DIR = "/home/ubuntu/work_dir/rf-detr/aug/mask"
os.makedirs(OUT_IMG_DIR, exist_ok=True)
os.makedirs(OUT_MASK_DIR, exist_ok=True)

NUM_PATCHES = 50  # số lượng ảnh crop muốn tạo
MIN_SIZE_RATIO = 0.3  # crop nhỏ nhất bằng 50% ảnh gốc
MAX_SIZE_RATIO = 0.55  # crop lớn nhất bằng 70% ảnh gốc


# ============================
# Augmentation Function
# ============================
def apply_augment(image, mask):
    aug_type = random.choice(["none", "flip_h", "flip_v", "rotate", "shear"])

    if aug_type == "flip_h":
        image = ImageOps.mirror(image)
        mask = ImageOps.mirror(mask)

    elif aug_type == "flip_v":
        image = ImageOps.flip(image)
        mask = ImageOps.flip(mask)

    elif aug_type == "rotate":
        angle = random.choice([90, 180, 270])
        image = image.rotate(angle, expand=True)
        mask = mask.rotate(angle, expand=True)

    # elif aug_type == "shear":
    #     shear_factor = random.uniform(-0.3, 0.3)  # độ xiên
    #     w, h = image.size
    #     matrix = (1, shear_factor, 0, 0, 1, 0)

    #     image = image.transform((w, h), Image.AFFINE, matrix, resample=Image.BICUBIC)
    #     mask = mask.transform((w, h), Image.AFFINE, matrix, resample=Image.NEAREST)

    return image, mask


# ============================
# Random Crop function
# ============================
MIN_CROP_SIZE = 560


def random_crop(image, mask):
    W, H = image.size

    # scale dựa theo min_size_ratio, nhưng cũng đảm bảo >= 560px
    scale = random.uniform(MIN_SIZE_RATIO, MAX_SIZE_RATIO)
    new_w = int(W * scale)
    new_h = int(H * scale)

    # 🔥 đảm bảo crop >= 560
    new_w = max(new_w, MIN_CROP_SIZE)
    new_h = max(new_h, MIN_CROP_SIZE)

    # 🔥 đảm bảo crop không vượt ảnh gốc
    new_w = min(new_w, W)
    new_h = min(new_h, H)

    # nếu crop size >= ảnh => crop full ảnh
    if new_w >= W or new_h >= H:
        return image.copy(), mask.copy()

    # chọn vị trí ngẫu nhiên
    x = random.randint(0, W - new_w)
    y = random.randint(0, H - new_h)

    cropped_img = image.crop((x, y, x + new_w, y + new_h))
    cropped_mask = mask.crop((x, y, x + new_w, y + new_h))

    return cropped_img, cropped_mask


# ============================
# Mask Normalization: background=0, obj=255
# ============================
def normalize_mask(mask):
    mask_np = np.array(mask)
    mask_norm = np.where(mask_np > 0, 255, 0).astype(np.uint8)
    return Image.fromarray(mask_norm)


def invert_mask(mask):
    mask_np = np.array(mask)
    # Đảo màu: 0 → 255, >0 → 0
    mask_inv = np.where(mask_np > 0, 0, 255).astype(np.uint8)
    return Image.fromarray(mask_inv)


# ============================
# Generate augmented data
# ============================
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--invert",
    type=bool,
    default=False,
    help="Choose to invert mask if mask of objects are black",
)
args = parser.parse_args()
invert = args.invert


def generate_augmented_data(img_path, mask_path):
    img = Image.open(img_path).convert("L")
    mask = Image.open(mask_path).convert("L")  # mask is grayscale

    W, H = img.size
    counter = 0
    for i in range(NUM_PATCHES):
        # 1) Clone để augment riêng biệt
        aug_img, aug_mask = apply_augment(img.copy(), mask.copy())

        # 2) Crop ngẫu nhiên
        crop_img, crop_mask = random_crop(aug_img, aug_mask)

        if crop_img is None:
            continue

        if invert:
            crop_mask = invert_mask(crop_mask)

        # 3) Lưu file
        img_out = f"{OUT_IMG_DIR}/aug_{counter}_{os.path.basename(img_path)}"
        mask_out = f"{OUT_MASK_DIR}/aug_{counter}_{os.path.basename(mask_path)}"

        crop_img.save(img_out)
        crop_mask.save(mask_out)

        print(f"[OK] Saved: {img_out}, {mask_out}")
        counter += 1


for img_path in IMG_PATHS:
    img_basename = os.path.basename(img_path)
    img_basename_noext = os.path.splitext(img_basename)[0]
    # print(img_basename_noext)

    for mask_path in MASK_PATHS:
        mask_basename = os.path.basename(mask_path)
        mask_basename_noext = os.path.splitext(mask_basename)[0]
        if img_basename_noext + "_mask" == mask_basename_noext:
            # if mask_basename.startswith(img_basename_noext):
            print("[OK] Match:", img_basename, "<->", os.path.basename(mask_path))
            generate_augmented_data(img_path, mask_path)
