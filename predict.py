from rfdetr import RFDETRSegPreview
import numpy as np
import cv2
import os

model = RFDETRSegPreview(
    pretrain_weights="./pretrained_checkpoint/checkpoint0149.pth"
)

dir = "./Input"
img_paths = [os.path.join(dir, img_name) for img_name in os.listdir(dir)]

output = "Output"
compare = "Compare"
output = compare
os.makedirs(output, exist_ok=True)

images = []
brightness_factor = 0.6
for img_path in img_paths:
    basename = os.path.basename(img_path)
    image = cv2.imread(img_path, cv2.COLOR_GRAY2BGR)
    darker = np.clip(image * brightness_factor, 0, 255).astype(np.uint8)
    image = darker
    images.append(image)
    if len(images) == 49: break

import time
start = time.time()
results = model.predict(images)
end = time.time()
print(f"Processed in {end - start:.2f} seconds")

for i, detections in enumerate(results):

    masks = detections.mask
    bboxes = detections.xyxy
    scores = detections.confidence
    class_ids = detections.class_id

    num_instances = masks.shape[0]
    h, w = image.shape[:2]
    overlay = image.copy()

    for i in range(num_instances):
        mask_bool = masks[i]  # (H, W)
        mask = mask_bool.astype(np.uint8)

        # tô màu (xanh lá) cho những điểm mask == 1
        overlay[mask == 1] = [0, 255, 0]

    # alpha blend 40%
    blended = cv2.addWeighted(overlay, 0.4, image, 0.6, 0)

    if blended.shape[0] != image.shape[0]:
        blended = cv2.resize(blended, (w, h))

    combined = np.concatenate([image, blended], axis=1)

    # ====== Lưu file ======
    out_file = os.path.join(output, f"vis_{basename}")
    cv2.imwrite(out_file, combined)
    print("Saved:", out_file)

