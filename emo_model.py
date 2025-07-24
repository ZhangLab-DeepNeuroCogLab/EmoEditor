import torch.nn as nn

class EmoDirectionEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.fc = nn.Sequential(
            nn.Linear(8, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 768),
            nn.ReLU(inplace=True),
            nn.Linear(768, 77 * 768)
        )


    def forward(self, x):
        
        x = self.fc(x)
        x = x.view(-1, 77, 768)
       
        return x
    
    