import torch.nn as nn
import torch
import random
import io, os
import numpy as np
from PIL import Image
from decompress_model import Adaptive_CS
import torchvision.transforms as transforms
from pre_process_image import process_image, init_model, generate_grad_cam_overlay

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class PrepNetwork(nn.Module):
    def __init__(self, color):
        self.color = color
        if color:
            m1_input_layer = 3
        if not color:
            m1_input_layer = 1
        super(PrepNetwork, self).__init__()
        self.initialP3 = nn.Sequential(
            nn.Conv2d(m1_input_layer, 50, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(50, 50, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(50, 50, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(50, 50, kernel_size=3, padding=1),
            nn.ReLU())
        self.initialP4 = nn.Sequential(
            nn.Conv2d(m1_input_layer, 50, kernel_size=4, padding=1),
            nn.ReLU(),
            nn.Conv2d(50, 50, kernel_size=4, padding=2),
            nn.ReLU(),
            nn.Conv2d(50, 50, kernel_size=4, padding=1),
            nn.ReLU(),
            nn.Conv2d(50, 50, kernel_size=4, padding=2),
            nn.ReLU())
        self.initialP5 = nn.Sequential(
            nn.Conv2d(m1_input_layer, 50, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv2d(50, 50, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv2d(50, 50, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv2d(50, 50, kernel_size=5, padding=2),
            nn.ReLU())
        self.finalP3 = nn.Sequential(
            nn.Conv2d(150, 50, kernel_size=3, padding=1),
            nn.ReLU())
        self.finalP4 = nn.Sequential(
            nn.Conv2d(150, 50, kernel_size=4, padding=1),
            nn.ReLU(),
            nn.Conv2d(50, 50, kernel_size=4, padding=2),
            nn.ReLU())
        self.finalP5 = nn.Sequential(
            nn.Conv2d(150, 50, kernel_size=5, padding=2),
            nn.ReLU())

        self.se1 = SEModule(50,2)
        self.se2 = SEModule(50,2)
        self.se3 = SEModule(50,2)

    def forward(self, p):
        p1 = self.initialP3(p)
        p2 = self.initialP4(p)
        p3 = self.initialP5(p)
        mid = torch.cat((p1, p2, p3), 1)
        p4 = self.finalP3(mid)
        p5 = self.finalP4(mid)
        p6 = self.finalP5(mid)

        p4 = self.se1(p4)
        p5 = self.se2(p5)
        p6 = self.se3(p6)

        out = torch.cat((p4, p5, p6), 1)
        return out

class HidingNetwork(nn.Module):
    def __init__(self,color, small,res=True,lambda_net=0.8):
        super(HidingNetwork, self).__init__()
        self.res = res
        self.lambda_net = lambda_net
        if color:
            if small:
                m2_input_layer = 4
            if not small:
                m2_input_layer = 153
            output_layer = 3
        if not color:
            if small:
                m2_input_layer = 2
            if not small:
                m2_input_layer = 151 #because cover is also black and white
            output_layer = 1
        self.initialH3 = nn.Sequential(
            nn.Conv2d(m2_input_layer, 50, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(50, 50, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(50, 50, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(50, 50, kernel_size=3, padding=1),
            nn.ReLU())
        self.initialH4 = nn.Sequential(
            nn.Conv2d(m2_input_layer, 50, kernel_size=4, padding=1),
            nn.ReLU(),
            nn.Conv2d(50, 50, kernel_size=4, padding=2),
            nn.ReLU(),
            nn.Conv2d(50, 50, kernel_size=4, padding=1),
            nn.ReLU(),
            nn.Conv2d(50, 50, kernel_size=4, padding=2),
            nn.ReLU())
        self.initialH5 = nn.Sequential(
            nn.Conv2d(m2_input_layer, 50, kernel_size=5, padding=2),
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
            nn.Conv2d(150, output_layer, kernel_size=1, padding=0))


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


class RevealNetwork(nn.Module):
    def __init__(self,small, color, res=True,lambda_net =0.8):
        super(RevealNetwork, self).__init__()
        self.res = res
        self.lambda_net = lambda_net
        self.color = color
        self.small = small
        if self.color==True:
            in_c = 3
        else:
            in_c = 1
        self.initialR3 = nn.Sequential(
            nn.Conv2d(in_c, 50, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(50, 50, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(50, 50, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(50, 50, kernel_size=3, padding=1),
            nn.ReLU())
        self.initialR4 = nn.Sequential(
            nn.Conv2d(in_c, 50, kernel_size=4, padding=1),
            nn.ReLU(),
            nn.Conv2d(50, 50, kernel_size=4, padding=2),
            nn.ReLU(),
            nn.Conv2d(50, 50, kernel_size=4, padding=1),
            nn.ReLU(),
            nn.Conv2d(50, 50, kernel_size=4, padding=2),
            nn.ReLU())
        self.initialR5 = nn.Sequential(
            nn.Conv2d(in_c, 50, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv2d(50, 50, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv2d(50, 50, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv2d(50, 50, kernel_size=5, padding=2),
            nn.ReLU())
        self.finalR3 = nn.Sequential(
            nn.Conv2d(150, 50, kernel_size=3, padding=1),
            nn.ReLU())
        self.finalR4 = nn.Sequential(
            nn.Conv2d(150, 50, kernel_size=4, padding=1),
            nn.ReLU(),
            nn.Conv2d(50, 50, kernel_size=4, padding=2),
            nn.ReLU())
        self.finalR5 = nn.Sequential(
            nn.Conv2d(150, 50, kernel_size=5, padding=2),
            nn.ReLU())
        self.finalR = nn.Sequential(
            nn.Conv2d(150, 3, kernel_size=1, padding=0))

    def forward(self, r,cover):
        if self.res:
            r = (r - self.lambda_net*cover)/(1-self.lambda_net)
        else:
            r = torch.cat((r,cover),dim=1)
        r1 = self.initialR3(r)
        r2 = self.initialR4(r)
        r3 = self.initialR5(r)
        mid = torch.cat((r1, r2, r3), 1)
        r4 = self.finalR3(mid)
        r5 = self.finalR4(mid)
        r6 = self.finalR5(mid)
        mid2 = torch.cat((r4, r5, r6), 1)
        out = self.finalR(mid2)
        return out

class SEModule(nn.Module):
    def __init__(self, channels, reduction, concat=False):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1, padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        module_input = x
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return module_input * x

def get_saliency(secret):
    
    mean_tensor_denorm = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1).to(device)
    std_tensor_denorm = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1).to(device)

    pre_model = init_model().to(device)
    processed_secrets_list = []
    norm_transform_for_cropped = transforms.Compose([
        transforms.Resize((secret.shape[2], secret.shape[3])), 
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    
    for i in range(secret.size(0)):
        single_secret_normalized_tensor = secret[i]
        
        single_secret_denormalized_tensor = single_secret_normalized_tensor * std_tensor_denorm + mean_tensor_denorm
        
        original_pil_img_for_processing = transforms.ToPILImage()(single_secret_denormalized_tensor.cpu().clamp(0,1))
        
        overlay_img, raw_heatmap_for_cropping = generate_grad_cam_overlay(pre_model, original_pil_img_for_processing)
        
        heatmap_pil_img = Image.fromarray((raw_heatmap_for_cropping * 255).astype(np.uint8))
        
        single_saliency_tensor = norm_transform_for_cropped(heatmap_pil_img).to(device)
        if single_saliency_tensor.ndim == 2: 
            single_saliency_tensor = single_saliency_tensor.unsqueeze(0) 
        elif single_saliency_tensor.ndim == 4: 
            single_saliency_tensor = single_saliency_tensor.squeeze(0) 
        elif single_saliency_tensor.ndim == 3 and single_saliency_tensor.shape[0] != 1:
            single_saliency_tensor = single_saliency_tensor[0:1, :, :]
        processed_secrets_list.append(single_saliency_tensor) 
    saliency_map_batch_concatenated = torch.cat(processed_secrets_list, dim=0) 
    saliency_map_final = saliency_map_batch_concatenated.unsqueeze(1) 
    
    return saliency_map_final

# Join three networks in one module
class Net(nn.Module):
    def __init__(self,base_project_path, resen=True,resde=True,lambda_net=0.8, small=True, compress =True, encrypt = True, color = False, add_cover=False):
        super(Net, self).__init__()
        self.resen = resen
        self.resde = resde
        self.lambda_net = lambda_net
        self.small = small 
        self.color = color
        self.encrypt = encrypt
        self.add_cover = add_cover
        self.compress = compress
        self.m1 = PrepNetwork(color)
        self.m2 = HidingNetwork(color, small,resen, lambda_net)
        self.m3 = RevealNetwork(small, color, resde,lambda_net)
        self.act = nn.Sigmoid()

        if self.compress:
            epoch = 320
            self.N_p = 13
            B = 8
            N = B*B

            model = Adaptive_CS(base_project_path, self.N_p, B, 1, torch.zeros(N, N))
            model = torch.nn.DataParallel(model).to(device)
            model_dir = '%s/layer_%d_block_%d' % (os.path.join(base_project_path,'compress_Data/model'), self.N_p, B)
            print("DEBUG: trying to load model from: ", model_dir)
            model.load_state_dict(torch.load('%s/net_params_%d.pkl' % (model_dir, epoch), map_location=device))
            self.model = model
        


    def forward(self, secret, cover, train = True):
        if self.encrypt:
            if self.small:
                secret_sal=get_saliency(secret)
                if not self.add_cover:
                    mid = torch.cat((secret, secret_sal), 1)
                if self.add_cover:
                    mid = torch.cat((secret, secret_sal, cover), 1)
            if not self.small:
                x_1 = self.m1(secret)
                mid = torch.cat((x_1, cover), 1)

            x_2_before_quan = self.m2(mid,cover) 
            x_2_after_quan = self.quan(x_2_before_quan,type='noise',train=train)

        if self.compress:
            x_2_compressed=[]
            for i in range(x_2_after_quan.shape[0]):
                x_2_single = x_2_after_quan[i:i+1]
                PhiT_Phi_basic, PhiT_y_basic, L, shape_info = model(
                x_2_single, 
                int(np.ceil(0.01 * N)),
                0.5,
                0.01,
                )   
                x_2_single_compressed=[PhiT_Phi_basic, PhiT_y_basic,L,shape_info]
                x_2_compressed.append(x_2_single_compressed)
            x_2_after_quan = x_2_compressed
            
            reconstructed_images = []
            rand_modes = [random.randint(0, 7) for _ in range(self.N_p)]
            for i in x_2_after_quan:
                x_2_single = self.model(
                        rand_modes,
                        i
                    )
                reconstructed_images.append(x_2_single)
            x_2= torch.cat(reconstructed_images, dim=0)
        else:
            x_2 = x_2_after_quan

        if self.encrypt:
            x_3 = self.m3(x_2,cover)
        else:
            x_3 + None
            
        return x_2, x_3

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
