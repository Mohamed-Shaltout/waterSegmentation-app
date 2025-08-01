import torch
import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from model import UNet  # or your custom model class

import tifffile as tiff  # Add this
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2

def preprocess_image(path):
    image = tiff.imread(path).astype(np.float32)  # ← USE TIFFFILE INSTEAD OF CV2

    # Resize to 512×512 if needed
    if image.shape[0] != 256 or image.shape[1] != 256:
        import cv2
        image = cv2.resize(image, (256, 256))

    # Keep only first 7 channels (or adjust depending on training)
    if image.shape[-1] > 7:
        selected_channels = [9, 1, 2, 3, 7, 10, 11] 
        image = image[:, :, selected_channels]

    transform = A.Compose([
        A.Normalize(mean=0.0, std=1.0, max_pixel_value=255.0),
        ToTensorV2()
    ])

    return transform(image=image)['image']



def postprocess_mask(mask, save_path):
    mask = (mask > 0.5).astype(np.uint8) * 255
    cv2.imwrite(save_path, mask)

def load_model(path, device):
    model = UNet(in_channels=7, out_channels=1)  # Update to match your model
    model.load_state_dict(torch.load(path, map_location=device))
    return model.to(device)
