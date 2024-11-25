import os
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from torch.utils.data.dataset import Dataset
import torch
from PIL import Image
import cv2
import numpy as np

class CamSegDataset(Dataset):
    """
    PyTorch CamSeg Dataset for loading and transforming images and masks for segmentation tasks.

    Args:
        images_dir (str): Path to the images folder.
        masks_dir (str): Path to the segmentation masks folder.
        classes (list): Class names to extract from segmentation mask.
        augmentation (albumentations.Compose): Data transformation pipeline (e.g., flip, scale).
        preprocessing (albumentations.Compose): Data preprocessing (e.g., normalization).
    """

    CLASSES = ['sky', 'building', 'pole', 'road', 'pavement', 
               'tree', 'signsymbol', 'fence', 'car', 
               'pedestrian', 'bicyclist', 'unlabelled']

    def __init__(self, images_dir, masks_dir, classes=None, augmentation=None, preprocessing=None):
        self.images_fps = sorted([os.path.join(images_dir, img) for img in os.listdir(images_dir)])
        self.masks_fps = sorted([os.path.join(masks_dir, img) for img in os.listdir(masks_dir)])
        # Convert class names to class indices
        self.class_values = [self.CLASSES.index(cls.lower()) for cls in classes]
        
        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def __getitem__(self, i):
        # Load image and mask
        image = cv2.imread(self.images_fps[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.masks_fps[i], 0)

        # Extract specific classes from mask
        masks = [(mask == v) for v in self.class_values]
        mask = np.stack(masks, axis=-1).astype('float32')

        # Add background if the mask is not binary
        if mask.shape[-1] != 1:
            background = 1 - mask.sum(axis=-1, keepdims=True)
            mask = np.concatenate((mask, background), axis=-1)

        # Apply augmentations
        if self.augmentation:
            augmented = self.augmentation(image=image, mask=mask)
            image, mask = augmented['image'], augmented['mask']

        # Apply preprocessing
        if self.preprocessing:
            processed = self.preprocessing(image=image, mask=mask)
            image, mask = processed['image'], processed['mask']

        # Convert to torch tensors
        image = transforms.ToTensor()(image)
        mask = torch.tensor(mask).permute(2, 0, 1)  # Move channels to first dimension

        return image, mask

    def __len__(self):
        return len(self.images_fps)