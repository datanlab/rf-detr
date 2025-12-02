from rfdetr import RFDETRSegPreview
import numpy as np
import cv2
import os

model = RFDETRSegPreview(
    pretrain_weights="/home/ubuntu/work_dir/rf-detr/run/checkpoint.pth"
)

dir = "/home/ubuntu/work_dir/rf-detr/Input"
img_paths = [os.path.join(dir, img_name) for img_name in os.listdir(dir)]

output = "Output"
compare = "Compare"
output = compare
os.makedirs(output, exist_ok=True)

for img_path in img_paths:
    basename = os.path.basename(img_path)
    image = cv2.imread(img_path, cv2.COLOR_GRAY2BGR)
    detections = model.predict(image)

    print(detections)

    masks = detections.mask
    bboxes = detections.xyxy
    scores = detections.confidence
    class_ids = detections.class_id

    num_instances = masks.shape[0]
    print("Num masks:", num_instances)

    # for i in range(num_instances):
    #     mask_bool = masks[i]  # shape (H, W), bool
    #     mask = mask_bool.astype(np.uint8) * 255

    #     out_path = f"./{output}/mask_{basename}"
    #     cv2.imwrite(out_path, mask)

    #     print("Saved:", out_path)
    h, w = image.shape[:2]
    overlay = image.copy()

    for i in range(num_instances):
        mask_bool = masks[i]  # (H, W)
        mask = mask_bool.astype(np.uint8)

        # tô màu (xanh lá) cho những điểm mask == 1
        overlay[mask == 1] = [0, 255, 0]

    # alpha blend 40%
    blended = cv2.addWeighted(overlay, 0.4, image, 0.6, 0)

    # ====== Ghép ảnh trái-phải ======
    # Resize overlay cho cùng chiều cao ảnh input
    if blended.shape[0] != image.shape[0]:
        blended = cv2.resize(blended, (w, h))

    combined = np.concatenate([image, blended], axis=1)

    # ====== Lưu file ======
    out_file = os.path.join(output, f"vis_{basename}")
    cv2.imwrite(out_file, combined)
    print("Saved:", out_file)
