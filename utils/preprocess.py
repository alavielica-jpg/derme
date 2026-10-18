import cv2
from PIL import Image
from torchvision import transforms

from utils.config import IMAGENET_MEAN, IMAGENET_STD, IMG_SIZE

def get_eval_transform(img_size=IMG_SIZE):
    return transforms.Compose([
        transforms.Resize(img_size),          
        transforms.ToTensor(),                
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def get_train_transform(img_size=IMG_SIZE):
    return transforms.Compose([
        transforms.Resize(img_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def bgr_to_model_input(bgr, img_size=IMG_SIZE):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)           
    pil = Image.fromarray(rgb)                            
    x = get_eval_transform(img_size)(pil)                 
    return x.unsqueeze(0)                                 