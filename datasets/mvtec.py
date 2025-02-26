import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T
from functools import lru_cache


# URL = 'ftp://guest:GU.205dldo@ftp.softronics.ch/mvtec_anomaly_detection/mvtec_anomaly_detection.tar.xz'
CLASS_NAMES = ['bottle', 'cable', 'capsule', 'carpet', 'grid',
               'hazelnut', 'leather', 'metal_nut', 'pill', 'screw',
               'tile', 'toothbrush', 'transistor', 'wood', 'zipper', 'coffee']

class LetterboxResize:
    def __init__(self, size, fill=128):
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

class MVTecDataset(Dataset):
    def __init__(self, dataset_path='D:/dataset/mvtec_anomaly_detection', class_name='bottle',
                 is_train=True, resize=256, cropsize=256):
        assert class_name in CLASS_NAMES, f'class_name: {class_name}, should be in {CLASS_NAMES}'
        self.dataset_path = dataset_path
        self.class_name = class_name
        self.is_train = is_train
        self.resize = resize
        self.cropsize = cropsize

        # 加载数据路径
        self.x, self.y, self.mask = self.load_dataset_folder()

        # 定义变换
        self.transform_x = T.Compose([
            LetterboxResize(self.resize),
            # T.CenterCrop(self.cropsize),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.transform_mask = T.Compose([
            T.Resize(self.resize, Image.Resampling.LANCZOS),
            T.CenterCrop(self.cropsize),
            T.ToTensor()
        ])

        # 初始化缓存 (限制大小)
        self.cache = {}
        self.transform_cache_size = 128

    @lru_cache(maxsize=256)  # 使用路径作为缓存键，避免重复 I/O
    def _load_image(self, path):
        return Image.open(path).convert('RGB')

    @lru_cache(maxsize=256)  # 可选：如果需要缓存 mask 加载
    def _load_mask(self, path):
        return Image.open(path) if path else None

    def __getitem__(self, idx):
        # 优先检查缓存
        if idx in self.cache:
            return self.cache[idx]

        x_path, y, mask_path = self.x[idx], self.y[idx], self.mask[idx]

        # 加载并变换图像 (用缓存的 _load_image 加速)
        x = self._load_image(x_path)
        x = self.transform_x(x)

        # 处理 mask
        if mask_path:
            mask = self._load_mask(mask_path)
            mask = self.transform_mask(mask)
        else:
            mask = torch.zeros([1, self.cropsize, self.cropsize])

        # 缓存处理结果 (限制缓存大小)
        if len(self.cache) < self.transform_cache_size:
            self.cache[idx] = (x, y, mask)

        return x, y, mask

    def __len__(self):
        return len(self.x)

    def load_dataset_folder(self):
        phase = 'train' if self.is_train else 'test'
        x, y, mask = [], [], []

        img_dir = os.path.join(self.dataset_path, self.class_name, phase)
        gt_dir = os.path.join(self.dataset_path, self.class_name, 'ground_truth')

        img_types = sorted(os.listdir(img_dir))
        for img_type in img_types:
            img_type_dir = os.path.join(img_dir, img_type)
            if not os.path.isdir(img_type_dir):
                continue

            img_fpaths = sorted([os.path.join(img_type_dir, f)
                                 for f in os.listdir(img_type_dir)
                                 if f.endswith(('.jpg', '.png'))])
            x.extend(img_fpaths)
            y.extend([0] * len(img_fpaths))      # 无标签时为正常样本
            mask.extend([None] * len(img_fpaths))  # 无标签时填 None

        assert len(x) == len(y), 'number of x and y should be same'
        return x, y, mask



