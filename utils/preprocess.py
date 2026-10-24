# Every image goes through the same steps: resize -> tensor -> normalize.

import cv2
from PIL import Image
from torchvision import transforms

from utils import config  

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_eval_transform(img_size=config.IMG_SIZE):
    # For validation, test and the live app: no randomness.
    return transforms.Compose([
        transforms.Resize(img_size),          # squashes to 224x224 (
        transforms.ToTensor(),                
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def get_train_transform(img_size=config.IMG_SIZE):
    return transforms.Compose([
        # Random zoom/crop: the model must recognize a rash at different scales
        # and positions, which helps real phone photos.
        transforms.RandomResizedCrop(img_size, scale=(0.7, 1.0), ratio=(0.85, 1.18)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),      
        transforms.RandomRotation(25),
        transforms.ColorJitter(brightness=0.3, contrast=0.3),
        transforms.RandomApply([transforms.GaussianBlur(5, sigma=(0.1, 1.5))], p=0.2),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.1)),  # hides small patches
    ])


def bgr_to_model_input(bgr, img_size=config.IMG_SIZE):
    # OpenCV BGR image -> tensor of shape (1, 3, H, W), ready for the CNN.
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)            # OpenCV is BGR, the model wants RGB
    pil = Image.fromarray(rgb)                            # same image type training uses
    x = get_eval_transform(img_size)(pil)                 # resize + tensor + normalize
    return x.unsqueeze(0)                                 # add the batch dimension