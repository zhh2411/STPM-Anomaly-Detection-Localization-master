import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as T

class LetterboxResize:
    def __init__(self, size, fill=128):
        self.size = size
        self.fill = fill

    def __call__(self, img):
        w, h = img.size
        scale = min(self.size / w, self.size / h)
        new_w, new_h = int(w * scale), int(h * scale)
        img_resized = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        new_img = Image.new('RGB', (self.size, self.size), (self.fill, self.fill, self.fill))
        new_img.paste(img_resized, ((self.size - new_w) // 2, (self.size - new_h) // 2))
        return new_img
    
# 加载图像
img_path = r"C:\modelproject\COMPUTERVISION\anodet\anodet\data\mvtec_dataset\coffee\test\test\021.jpg"
img = Image.open(img_path).convert("RGB")

# 定义变换
resize = 256
cropsize = 256
# transform_x = T.Compose([
#     T.Resize(resize, Image.Resampling.LANCZOS),
#     T.CenterCrop(cropsize),
#     T.ToTensor(),
#     T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# ])
transform_x = T.Compose([
    LetterboxResize(256),  # 无拉伸缩放 + 自动填充                                      # 确保尺寸一致
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 可视化每一步
fig, axs = plt.subplots(1, 4, figsize=(20, 5))

# 1️⃣ 原图
axs[0].imshow(img)
axs[0].set_title("Original")
axs[0].axis("off")

# 2️⃣ Resize
resized_img = T.Resize(resize, Image.Resampling.LANCZOS)(img)
axs[1].imshow(resized_img)
axs[1].set_title("Resized")
axs[1].axis("off")

# 3️⃣ CenterCrop
cropped_img = T.CenterCrop(cropsize)(resized_img)
axs[2].imshow(cropped_img)
axs[2].set_title("CenterCrop")
axs[2].axis("off")

# 4️⃣ ToTensor + Normalize (反归一化后可视化)
def denormalize(tensor, mean, std):
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor

tensor_img = transform_x(img)  # (C, H, W)
img_denorm = denormalize(tensor_img.clone(), [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
axs[3].imshow(img_denorm.permute(1, 2, 0).numpy())
axs[3].set_title("Normalized (Denorm for display)")
axs[3].axis("off")

plt.show()
