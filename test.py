import torch
from torchvision import transforms
import os
import numpy as np
import argparse
import random
from PIL import Image
from utils import cal_ssim, load_emo_predictor
from ip2p import InstructPix2Pix
from emo_model import EmoDirectionEncoder


if __name__ == '__main__':
   
    parser = argparse.ArgumentParser()    
    parser.add_argument('--random_seed', type=int, default=75863,
                        help="try a different seed or set it to `None` -- the model will generate one randomly.")
    parser.add_argument('--input_path', type=str, default="demo_img/demo_fire.png",
                        help="use your input image path")
    parser.add_argument('--target_emo', type=str, default="awe", 
                        choices=["amusement", "anger", "awe", "contentment", "disgust", "excitement", "fear", "sadness"])
    parser.add_argument('--test_save_dir', type=str, default="save_test_res")
    parser.add_argument('--num_inference_steps', type=int, default=30)
    parser.add_argument('--num_loop', type=int, default=30)
    args = parser.parse_args()

    classes = ["amusement", "anger", "awe", "contentment", "disgust", "excitement", "fear", "sadness"]
    positive = ["amusement", "awe", "contentment", "excitement"]
    negative = ["anger", "disgust", "fear", "sadness"]

    test_tf = transforms.Compose([
        transforms.Resize((224, 224)),         
        transforms.ToTensor(),
    ])

    os.makedirs(args.test_save_dir, exist_ok=True)
        
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

    # load model
    enc_model_dir = "model/model_emoenc.pt"
    ip2p_model_dir = "model/model_ip2p.pt"
    
    emo_enc = EmoDirectionEncoder().to(device)
    emo_enc.load_state_dict(torch.load(enc_model_dir, map_location=device))
    emo_enc.eval()
            
    ip2p_model = InstructPix2Pix(device).to(device).half()
    ip2p_model.load_state_dict(torch.load(ip2p_model_dir, map_location=device))
    ip2p_model.requires_grad_(False)
    ip2p_model.eval()
    
    emo_preditor = load_emo_predictor(save_dir="emo_predictor/emo_predictor.pt").to(device)
    emo_preditor.requires_grad_(False)
    emo_preditor.eval()

    img_source = Image.open(args.input_path).convert("RGB")
    # img_size = img_source.size
    img_source = test_tf(img_source).unsqueeze(0).to(device)

    src_emo_code = emo_preditor(img_source.float())
    src_softmax = torch.softmax(src_emo_code[0], dim=0).detach()

    tgt_label = classes.index(args.target_emo)
    tgt_onehot = np.zeros(8)
    tgt_onehot[tgt_label] = 1
    tgt_onehot = torch.from_numpy(tgt_onehot)
            
    pred_Img = [img_source[0]]
    best_tgt_prob = 0
    best_tgt_list = []
    best_same_prob = 0
    best_same_list = []
    
    with torch.no_grad():
        
        random_seed = args.random_seed if args.random_seed != None else random.randint(0, 100000)
        print('random_seed:', random_seed)
        torch.manual_seed(random_seed)
        torch.cuda.manual_seed(random_seed)
        np.random.seed(random_seed)
        
        tgt_onehot = tgt_onehot.to(device).float()   
        src_softmax = src_softmax.to(device).float()
        
        emo_direction = tgt_onehot - src_softmax
        emo_direction = emo_direction.to(device).float()
        encoder_hidden_states = emo_enc(emo_direction)
                
        edited_image = ip2p_model(encoder_hidden_states.to(device).half(),
                        img_source.to(device).half(),
                        img_source.to(device).half(),
                        diffusion_steps=args.num_inference_steps)
        
        predict_emo_code = emo_preditor(edited_image.float())
        predict_emo = predict_emo_code.argmax(dim=1)
        predict_label = predict_emo[0].item()
        predict_softmax = torch.softmax(predict_emo_code[0], dim=0).detach()
        
        ssim_pred = cal_ssim(img_source[0].mul(255).clamp(0, 255).byte().permute(1, 2, 0).data.cpu().numpy(), \
                                    edited_image[0].mul(255).clamp(0, 255).byte().permute(1, 2, 0).data.cpu().numpy())
        if (ssim_pred < 0.8 and ssim_pred > 0.5) or predict_label == tgt_label:
            
            if classes[predict_label] in positive and classes[tgt_label] in positive:
                predict_emo_prob = torch.softmax(predict_emo_code[0], dim=0)
                confidence = predict_emo_prob[predict_label]
                
                if confidence > best_same_prob:
                    best_same_prob = confidence.item()
                    best_same_list.append(edited_image[0])
                
            elif classes[predict_label] in negative and classes[tgt_label] in negative:
                
                predict_emo_prob = torch.softmax(predict_emo_code[0], dim=0)
                confidence = predict_emo_prob[predict_label]
                
                if confidence > best_same_prob:
                    best_same_prob = confidence.item()
                    best_same_list.append(edited_image[0])
            
            
            predict_emo_prob = torch.softmax(predict_emo_code[0], dim=0)
            confidence = predict_emo_prob[tgt_label]
            
            if confidence > best_tgt_prob:
                best_tgt_prob = confidence.item()
                best_tgt_list.append(edited_image[0])
                    
            pred_Img.append(edited_image[0])
        else:
            edited_image = pred_Img[-1].unsqueeze(0)

        while best_tgt_prob < 0.6 and len(pred_Img) < args.num_loop:
        
            emo_direction = (tgt_onehot - predict_softmax).float()
            encoder_hidden_states = emo_enc(emo_direction)
            edited_image = ip2p_model(encoder_hidden_states.to(device).half(),
                                    edited_image.to(device).half(),
                                    img_source.to(device).half(),
                                    diffusion_steps=args.num_inference_steps)
            
            predict_emo_code = emo_preditor(edited_image.float())
            predict_emo = predict_emo_code.argmax(dim=1)    
            predict_label = predict_emo[0].item()
            
            ssim_pred = cal_ssim(img_source[0].mul(255).clamp(0, 255).byte().permute(1, 2, 0).data.cpu().numpy(), \
                                    edited_image[0].mul(255).clamp(0, 255).byte().permute(1, 2, 0).data.cpu().numpy())
            if (ssim_pred < 0.8 and ssim_pred > 0.5) or predict_label == tgt_label:
                if classes[predict_label] in positive and classes[tgt_label] in positive:
                    predict_emo_prob = torch.softmax(predict_emo_code[0], dim=0)
                    confidence = predict_emo_prob[predict_label]
                    
                    if confidence > best_same_prob:
                        best_same_prob = confidence.item()
                        best_same_list.append(edited_image[0])
                    
                elif classes[predict_label] in negative and classes[tgt_label] in negative:
                    
                    predict_emo_prob = torch.softmax(predict_emo_code[0], dim=0)
                    confidence = predict_emo_prob[predict_label]
                    
                    if confidence > best_same_prob:
                        best_same_prob = confidence.item()
                        best_same_list.append(edited_image[0])
                
                
                predict_emo_prob = torch.softmax(predict_emo_code[0], dim=0)
                confidence = predict_emo_prob[tgt_label]
                
                if confidence > best_tgt_prob:
                    best_tgt_prob = confidence.item()
                    best_tgt_list.append(edited_image[0])
                        
                pred_Img.append(edited_image[0])
            else:
                edited_image = pred_Img[-1].unsqueeze(0)      
        
        if predict_label != tgt_label:
            if best_tgt_prob > 0.6:
                pred_Img.append(best_tgt_list[-1])
            elif best_same_prob != 0:
                pred_Img.append(best_same_list[-1])
            elif best_tgt_prob != 0:
                pred_Img.append(best_tgt_list[-1])
            
        img_save = Image.fromarray(pred_Img[-1].mul(255).clamp(0, 255).byte().permute(1, 2, 0).data.cpu().numpy())
        # img_save = img_save.resize(img_size)
        img_save_pth = os.path.join(args.test_save_dir, str(random_seed)+"_"+args.target_emo+"_"+args.input_path.split("/")[-1])
        img_save.save(img_save_pth)

        print(f"Result saved in {img_save_pth}")


       