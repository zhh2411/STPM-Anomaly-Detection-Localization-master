import numpy as np
import json
from pycocotools import mask
import cv2
import torchvision.transforms as T
from PIL import Image
import matplotlib.pyplot as plt

# 解析COCO格式标注数据
coco_annotations = {
    "licenses": [{"name": "", "id": 0, "url": ""}],
    "info": {"contributor": "", "date_created": "", "description": "", "url": "", "version": "", "year": ""},
    "categories": [{"id": 1, "name": "mask", "supercategory": ""}],
    "images": [
        {
            "id": 1,
            "width": 2304,
            "height": 1296,
            "file_name": "360_Camera_0101_20250217_141643.mp4_frame_20250217144019.jpg",
            "license": 0,
            "flickr_url": "",
            "coco_url": "",
            "date_captured": 0
        }
    ],
    "annotations": [
        {
            "id": 1,
            "image_id": 1,
            "category_id": 1,
            "segmentation": [[196.46,692.44,257.47,712.22,356.4,801.26,631.76,1058.48,890.63,1292.62,0.25,1294.43,1.9,835.89,82.69,743.55,138.75,707.28,165.13,695.73,165.13,695.73]],
            "area": 327248.0,
            "bbox": [0.25, 692.44, 890.38, 601.99],
            "iscrowd": 0,
            "attributes": {"occluded": False}
        },
        {
            "id": 2,
            "image_id": 1,
            "category_id": 1,
            "segmentation": [[3.54,625.0,18.38,601.91,6.84,580.31,81.04,502.82,53.01,464.89,128.86,389.21,120.61,369.26,133.8,344.53,114.02,298.36,26.63,191.18,257.47,5.03,0.25,0.08,1.9,324.91]],
            "area": 58092.0,
            "bbox": [0.25, 0.08, 257.22, 624.92],
            "iscrowd": 0,
            "attributes": {"occluded": False}
        },
        {
            "id": 3,
            "image_id": 1,
            "category_id": 1,
            "segmentation": [[2302.05,75.76,2191.58,267.2,2107.48,456.65,2013.5,662.92,1878.29,915.2,1685.38,1231.78,1644.16,1292.62,2304.0,1294.43,2304.0,796.88]],
            "area": 384903.0,
            "bbox": [1644.16, 75.76, 659.84, 1218.67],
            "iscrowd": 0,
            "attributes": {"occluded": False}
        }
    ]
}

class LetterboxResize:
    def __init__(self, size, fill=0):
        self.size = size
        self.fill = fill

    def __call__(self, img):
        w, h = img.size
        scale = min(self.size / w, self.size / h)
        new_w, new_h = int(w * scale), int(h * scale)
        img_resized = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        new_img = Image.new('RGB', (self.size, self.size), (self.fill,) * 3)
        new_img.paste(img_resized, ((self.size - new_w) // 2, (self.size - new_h) // 2))
        return new_img

def create_binary_mask(coco_annotations, save_path="binary_mask.png", display=True):
    image_info = coco_annotations["images"][0]
    width, height = image_info["width"], image_info["height"]

    mask_image = np.zeros((height, width), dtype=np.uint8)

    for ann in coco_annotations["annotations"]:
        segmentation = ann["segmentation"]
        for seg in segmentation:
            polygon = np.array(seg, dtype=np.int32).reshape((-1, 2))
            cv2.fillPoly(mask_image, [polygon], color=1)

    # 颜色反转
    mask_image = 1 - mask_image

    cv2.imwrite(save_path, mask_image * 255)

    mask_pil = Image.fromarray(mask_image * 255).convert('RGB')

    transform_mask = T.Compose([
        LetterboxResize(256),
        T.ToTensor()
    ])

    mask_tensor = transform_mask(mask_pil)

    transformed_mask_pil = T.ToPILImage()(mask_tensor)
    transformed_mask_pil.save("resized_" + save_path)

    if display:
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(mask_pil, cmap='gray')
        plt.title("origin")

        plt.subplot(1, 2, 2)
        plt.imshow(transformed_mask_pil)
        plt.title("Letterbox")

        plt.show()

    return mask_tensor, transformed_mask_pil

def load_and_process_mask(image_path):
    # 1. 读取图像
    mask_pil = Image.open(image_path).convert('L')  # 转为灰度
    
    # 3. 转换为 NumPy 数组
    mask_np = np.array(mask_pil, dtype=np.uint8)
    
    # 4. 归一化：将 255 变为 1，0 仍然是 0
    mask_np = (mask_np > 128).astype(np.uint8)  # 二值化处理

    return mask_np

if __name__ == '__main__':
    mask_tensor, mask_pil = create_binary_mask(coco_annotations)
    mask_array = load_and_process_mask('./resized_binary_mask.png')
    print(mask_array.shape)  # 应该是 (256, 256)
    print(mask_array)  # 0（黑色）和 1（白色）