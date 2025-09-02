import torch.nn as nn
import torch, os
from torchvision.utils import save_image
device = torch.device('cuda' if torch.cuda.is_available() else "cpu")


class HidingNetwork(nn.Module):
    def __init__(self, res=True,lambda_net=0.8):
        super(HidingNetwork, self).__init__()
        self.res = res
        self.lambda_net = lambda_net
        #print("DEBUG: output_layer = ", output_layer)
        self.initialH3 = nn.Sequential(
            nn.Conv2d(4, 50, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(50, 50, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(50, 50, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(50, 50, kernel_size=3, padding=1),
            nn.ReLU())
        self.initialH4 = nn.Sequential(
            nn.Conv2d(4, 50, kernel_size=4, padding=1),
            nn.ReLU(),
            nn.Conv2d(50, 50, kernel_size=4, padding=2),
            nn.ReLU(),
            nn.Conv2d(50, 50, kernel_size=4, padding=1),
            nn.ReLU(),
            nn.Conv2d(50, 50, kernel_size=4, padding=2),
            nn.ReLU())
        self.initialH5 = nn.Sequential(
            nn.Conv2d(4, 50, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv2d(50, 50, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv2d(50, 50, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv2d(50, 50, kernel_size=5, padding=2),
            nn.ReLU())
        self.finalH3 = nn.Sequential(
            nn.Conv2d(150, 50, kernel_size=3, padding=1),
            nn.ReLU())
        self.finalH4 = nn.Sequential(
            nn.Conv2d(150, 50, kernel_size=4, padding=1),
            nn.ReLU(),
            nn.Conv2d(50, 50, kernel_size=4, padding=2),
            nn.ReLU())
        self.finalH5 = nn.Sequential(
            nn.Conv2d(150, 50, kernel_size=5, padding=2),
            nn.ReLU())
        self.finalH = nn.Sequential(
            nn.Conv2d(150, 3, kernel_size=1, padding=0))


    def forward(self, h,cover): 
        h1 = self.initialH3(h) 
        h2 = self.initialH4(h) 
        h3 = self.initialH5(h) 
        mid = torch.cat((h1, h2, h3), 1)
        h4 = self.finalH3(mid)
        h5 = self.finalH4(mid)
        h6 = self.finalH5(mid)
        mid2 = torch.cat((h4, h5, h6), 1)
        out = self.finalH(mid2)

        if self.res:
             out = (1-self.lambda_net)*out + self.lambda_net*cover 
        return out


class Netw(nn.Module):
    def __init__(self, resen=True,resde=True,lambda_net=0.8):
        super(Netw, self).__init__()
        self.resen = resen
        self.resde = resde
        self.lambda_net = lambda_net
        self.act = nn.Sigmoid()
        self.device = device 
        
        self.m2 = HidingNetwork( resen, lambda_net).to(self.device)
        self.eval() 
        
    def forward(self, secret_rgb, saliency_map, cover_img, train=True, **args): 
        hiding_input_tensor = torch.cat((secret_rgb, saliency_map), 1)
        mix_img_before_quan = self.m2(hiding_input_tensor, cover_img)
        mix_img_after_quan = self.quan(mix_img_before_quan, type='noise', train=train)

        return mix_img_after_quan, saliency_map


    def quan(self,x,type='noise',train=True):
        if type=='round':
            x = torch.round(torch.clamp(x*255.,0,255.))/255.
        elif type == 'noise':
            x = x*255.
            if train:
                noise = torch.nn.init.uniform_(torch.zeros_like(x), -0.5, 0.5).to(x.device)
                output = x + noise
                output = torch.clamp(output, 0, 255.)
            else:
                output = x.round() * 1.0
                output = torch.clamp(output, 0, 255.)
        else:
            raise ValueError("quan is not implemented for this type.")
        return output/255.
    
