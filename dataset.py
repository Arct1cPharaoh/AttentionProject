import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import scipy.io
import cv2  # for Gaussian blur

class SaliconDataset(Dataset):
    def __init__(self, image_root, fixation_root, split, transform=None):
        self.image_dir = os.path.join(image_root, split)
        self.fixation_dir = os.path.join(fixation_root, split)
        self.image_files = sorted(os.listdir(self.image_dir))
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_name)
        img = Image.open(img_path).convert("RGB")

        mat_name = img_name.replace(".jpg", ".mat")
        mat_path = os.path.join(self.fixation_dir, mat_name)
        mat_data = scipy.io.loadmat(mat_path)

        # image resolution
        h, w = mat_data["resolution"][0]
        heatmap = np.zeros((h, w), dtype=np.float32)

        # add Gaussian blobs to each fixation point from all subjects
        for subj in mat_data["gaze"][0]:
            fixations = subj["fixations"]
            for x, y in fixations:
                if 0 <= int(y) < h and 0 <= int(x) < w:
                    heatmap[int(y), int(x)] += 1

        # normalize and blur
        heatmap = cv2.GaussianBlur(heatmap, (0, 0), sigmaX=10)
        heatmap = heatmap / heatmap.max() if heatmap.max() > 0 else heatmap
        sal = Image.fromarray((heatmap * 255).astype(np.uint8)).convert("L")

        if self.transform:
            img = self.transform(img)
            sal = self.transform(sal)

        return img, sal
