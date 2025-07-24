import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models
from skimage.metrics import structural_similarity as ssim


def load_emo_predictor(save_dir="emo_predictor/emo_predictor.pt"):    
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(in_features=512, out_features=8, bias=True)
    model.load_state_dict(torch.load(save_dir))
    model.eval()   
     
    return model


def cal_L1(img, img2):
    img = img.astype(np.float64)
    img2 = img2.astype(np.float64)
    loss = np.mean(np.abs(img - img2))
    return loss


def cal_ssim(img1, img2, win_size=51):
    if len(img1.shape) == 3 and img1.shape[2] == 3:
        img1_gray = cv2.cvtColor(img1.astype(np.uint8), cv2.COLOR_RGB2GRAY)
        img2_gray = cv2.cvtColor(img2.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    else:
        img1_gray = img1.astype(np.uint8)
        img2_gray = img2.astype(np.uint8)
    ssim_score = ssim(img1_gray, img2_gray, win_size=win_size)
    return ssim_score

