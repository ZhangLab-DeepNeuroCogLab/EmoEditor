import torch
from PIL import Image
from torchvision import transforms
import numpy as np
from scipy.stats import entropy
from pytorch_grad_cam import GradCAM
from utils import cal_L1, load_emo_predictor


#############################################################
image_name_src = "" # use your source image path
image_name_edit = "" # use your edit image path
tgt_emo = "" # one class from 8 classes
#############################################################


device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

emo_preditor = load_emo_predictor(save_dir="emo_predictor/emo_predictor.pt").to(device)
emo_preditor.eval()
target_layers = [emo_preditor.layer4[-1]]
cam = GradCAM(model=emo_preditor, target_layers=target_layers)

test_tf = transforms.Compose([
        transforms.Resize((224, 224)),         
        transforms.ToTensor(),
    ])

classes = ["amusement", "anger", "awe", "contentment", "disgust", "excitement", "fear", "sadness"]
eps = 1e-10


## cal S_emo 
tgt_idx = classes.index(tgt_emo)
target_emo = np.zeros(8)
target_emo[tgt_idx] = 1
                
src_img_PIL = Image.open(image_name_src).resize((224, 224)).convert('RGB')
src_img = test_tf(src_img_PIL).unsqueeze(0).to(device)
src_emo = emo_preditor(src_img)
src_emo_softmax = torch.softmax(src_emo[0], dim=0).detach().cpu().numpy()

edit_img_PIL = Image.open(image_name_edit).resize((224, 224)).convert('RGB')
edit_img = test_tf(edit_img_PIL).unsqueeze(0).to(device)
edit_emo = emo_preditor(edit_img)
edit_emo_softmax = torch.softmax(edit_emo[0], dim=0).detach().cpu().numpy()

kld_src = entropy(target_emo, src_emo_softmax)
kld_edit = entropy(target_emo, edit_emo_softmax)
S_emo = max(0, (kld_src - kld_edit)) / (kld_src + eps)

## cal S_str
src_grayscale_cam = cam(input_tensor=src_img)[0, :]
src_grayscale_cam = (src_grayscale_cam > 0.5).astype(np.uint8)
src_cam = np.repeat(src_grayscale_cam[:, :, np.newaxis], 3, axis=2)

src_img_np = np.array(src_img_PIL)
edit_img_np = np.array(edit_img_PIL)

src_img_emo_region = src_img_np * src_cam
edit_img_emo_region = edit_img_np * src_cam

L_emo = cal_L1(src_img_emo_region, edit_img_emo_region)
L_full = cal_L1(src_img_np, edit_img_np)
S_str = L_emo / (L_full + eps)

## cal ESMI
alpha = 0.5
ESMI = alpha * S_str +  (1 - alpha) * S_emo
print(f"{image_name_src} -> {image_name_edit}: {ESMI*100:.2f}")