import os
import random

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as TF


class LEVIRCD_Dataset(Dataset):
    """Load prepared bitemporal patches stored under A, B, and label folders."""

    def __init__(self, data_dir, mode="train"):
        super().__init__()
        if mode not in {"train", "val", "test"}:
            raise ValueError(f"Unsupported dataset split: {mode}")

        self.mode = mode
        self.A_dir = os.path.join(data_dir, mode, "A")
        self.B_dir = os.path.join(data_dir, mode, "B")
        self.label_dir = os.path.join(data_dir, mode, "label")

        for directory in (self.A_dir, self.B_dir, self.label_dir):
            if not os.path.isdir(directory):
                raise FileNotFoundError(f"Required dataset directory not found: {directory}")

        self.image_names = sorted(os.listdir(self.A_dir))
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, index):
        image_name = self.image_names[index]
        image_a = Image.open(os.path.join(self.A_dir, image_name)).convert("RGB")
        image_b = Image.open(os.path.join(self.B_dir, image_name)).convert("RGB")
        label = Image.open(os.path.join(self.label_dir, image_name)).convert("L")

        if self.mode == "train":
            image_a, image_b, label = self.apply_train_aug(
                image_a,
                image_b,
                label,
            )

        t1 = self.normalize(TF.to_tensor(image_a))
        t2 = self.normalize(TF.to_tensor(image_b))
        mask = (TF.to_tensor(label) > 0).float()
        return t1, t2, mask

    @staticmethod
    def apply_train_aug(image_a, image_b, label):
        if random.random() > 0.5:
            image_a = TF.hflip(image_a)
            image_b = TF.hflip(image_b)
            label = TF.hflip(label)

        if random.random() > 0.5:
            image_a = TF.vflip(image_a)
            image_b = TF.vflip(image_b)
            label = TF.vflip(label)

        angle = random.choice([0, 90, 180, 270])
        if angle > 0:
            image_a = TF.rotate(image_a, angle)
            image_b = TF.rotate(image_b, angle)
            label = TF.rotate(label, angle)

        return image_a, image_b, label

