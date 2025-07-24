import json
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
    
class EmoPairDataset(Dataset):
    def __init__(self, dataset_name, transform=None):

        self.data = []
        self.dataset_name = dataset_name
        self.transform = transform
        self.classes = ["amusement", "anger", "awe", "contentment", "disgust", "excitement", "fear", "sadness"]
        
        self.positive = ["amusement", "awe", "contentment", "excitement"]
        self.negative = ["anger", "disgust", "fear", "sadness"]
                    
        if self.dataset_name == "EmoPair":
            json_path = "./json/data_"+self.dataset_name+".json"            
            with open(json_path) as f:
                self.data = json.load(f)

            # # only use EPGS
            # self.data = [item for item in self.data if "EPGS" in item["img_src_path"]]

                    
        print("Samples:", len(self.data))

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        emo_src = item['class_source']
        emo_tgt = item['class_target']
        prompt = item['prompt']
        emo_trans = emo_src + " to " + emo_tgt

        img_src_path = "EmoPair/"+item['img_src_path']
        img_source = Image.open(img_src_path).convert("RGB")
        img_tgt_path = "EmoPair/"+item['img_tgt_path']
        img_target = Image.open(img_tgt_path).convert("RGB")
        if self.transform is not None:
            img_source = self.transform(img_source)
            img_target = self.transform(img_target)
            
        tgt_onehot = np.zeros(8)
        tgt_onehot[int(self.classes.index(emo_tgt))] = 1
        tgt_onehot = torch.from_numpy(tgt_onehot)                
        src_softmax = torch.from_numpy(np.array(item['source_softmax']))        
        emo_direction = tgt_onehot - src_softmax
                
        return prompt, emo_trans, img_source, img_target, emo_direction
            
        