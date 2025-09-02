import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from utils import H
import math, os
from collections import OrderedDict  
from get_prep import PrepNetwork as PNet
import kornia.color as kcolor
from mobilenet_v3 import mobilenet_v3



class CBR(nn.Module):
    '''
    This class defines the convolution layer with batch normalization and PReLU activation
    '''

    def __init__(self, nIn, nOut, kSize, stride=1, groups=1):
        '''

        :param nIn: number of input channels
        :param nOut: number of output channels
        :param kSize: kernel size
        :param stride: stride rate for down-sampling. Default is 1
        '''
        super().__init__()
        padding = int((kSize - 1) / 2)
        self.conv = nn.Conv2d(nIn, nOut, kSize, stride=stride, padding=padding, bias=False, groups=groups)
        self.bn = nn.BatchNorm2d(nOut)
        self.act = nn.PReLU(nOut)

    def forward(self, input):
        '''
        :param input: input feature map
        :return: transformed feature map
        '''
        output = self.conv(input)
        # output = self.conv1(output)
        output = self.bn(output)
        output = self.act(output)
        return output

# https://github.com/haochange/DUpsampling/blob/master/models/dunet.py
class DUpsampling(nn.Module):
    def __init__(self, inplanes, scale, num_class=21, pad=0):
        super(DUpsampling, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, num_class * scale * scale, kernel_size=1, padding = pad,bias=False)
        self.scale = scale
    
    def forward(self, x):
        x = self.conv1(x)
        N, C, H, W = x.size()

        # N, H, W, C
        x_permuted = x.permute(0, 2, 3, 1) 

        # N, H, W*scale, C/scale
        x_permuted = x_permuted.contiguous().view((N, H, W * self.scale, int(C / (self.scale))))

        # N, W*scale,H, C/scale
        x_permuted = x_permuted.permute(0, 2, 1, 3)
        # N, W*scale,H*scale, C/(scale**2)
        x_permuted = x_permuted.contiguous().view((N, W * self.scale, H * self.scale, int(C / (self.scale * self.scale))))

        # N,C/(scale**2),W*scale,H*scale
        x = x_permuted.permute(0, 3, 2, 1)
        
        return x

# classic residual block
class RB(nn.Module):
    def __init__(self, nf, bias, kz=3):
        super(RB, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(nf, nf, kz, padding=kz // 2, bias=bias), nn.ReLU(inplace=True),
            nn.Conv2d(nf, nf, kz, padding=kz // 2, bias=bias),
        )
        
    def forward(self, x):
        return x + self.body(x)

# proximal mapping network (https://github.com/cszn/DPIR)
class P(nn.Module):
    def __init__(self, in_nf, out_nf):
        super(P, self).__init__()
        bias, block, nb, scale_factor = False, RB, 2, 2
        mid_nf = [16, 32, 64, 128]

        conv = lambda in_nf, out_nf: nn.Conv2d(in_nf, out_nf, 3, padding=1, bias=bias)
        up = lambda nf, scale_factor: nn.ConvTranspose2d(nf, nf, scale_factor, stride=scale_factor, bias=bias)
        down = lambda nf, scale_factor: nn.Conv2d(nf, nf, scale_factor, stride=scale_factor, bias=bias)
        
        self.down1 = nn.Sequential(conv(in_nf, mid_nf[0]), *[block(mid_nf[0], bias) for _ in range(nb)], down(mid_nf[0], scale_factor))
        self.down2 = nn.Sequential(conv(mid_nf[0], mid_nf[1]), *[block(mid_nf[1], bias) for _ in range(nb)], down(mid_nf[1], scale_factor))
        self.down3 = nn.Sequential(conv(mid_nf[1], mid_nf[2]), *[block(mid_nf[2], bias) for _ in range(nb)], down(mid_nf[2], scale_factor))
        
        self.body  = nn.Sequential(conv(mid_nf[2], mid_nf[3]), *[block(mid_nf[3], bias) for _ in range(nb)], conv(mid_nf[3], mid_nf[2]))
        
        self.up3 = nn.Sequential(up(mid_nf[2], scale_factor), *[block(mid_nf[2], bias) for _ in range(nb)], conv(mid_nf[2], mid_nf[1]))
        self.up2 = nn.Sequential(up(mid_nf[1], scale_factor), *[block(mid_nf[1], bias) for _ in range(nb)], conv(mid_nf[1], mid_nf[0]))
        self.up1 = nn.Sequential(up(mid_nf[0], scale_factor), *[block(mid_nf[0], bias) for _ in range(nb)], conv(mid_nf[0], out_nf))

    def forward(self, x):
        x1 = self.down1(x)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x = self.body(x3)
        x = self.up3(x + x3)  # three skip connections for the last three scales
        x = self.up2(x + x2)
        x = self.up1(x + x1)
        return x

class Phase(nn.Module):
    def __init__(self, img_nf, B):
        super(Phase, self).__init__()
        #bias, nf, nb, onf = True, 8, 3, 3  # config of E

        self.rho = nn.Parameter(torch.Tensor([0.5]))
        self.P = P(img_nf, img_nf)  # input: [Z | saliency_feature]
        self.B = B  # default: 32
        #self.E = nn.Sequential(  # saliency feature extractor
        #    nn.Conv2d(1, nf, 1, bias=bias),
        #    *[RB(nf, bias, kz=1) for _ in range(nb)],
        #    nn.Conv2d(nf, onf, 1, bias=bias)
        #)
    
    #def forward(self, x, cs_ratio_map, PhiT_Phi, PhiT_y, mode, shape_info):
    def forward(self, x, PhiT_Phi, PhiT_y, mode, shape_info):
        b, l, h, w = shape_info
        
        # block gradient descent
        x = x - self.rho * (PhiT_Phi.matmul(x) - PhiT_y)
        
        # saliency information guided proximal mapping (with RTE strategy)
        x = x.reshape(b, l, -1).permute(0, 2, 1)
        x = F.fold(x, output_size=(h, w), kernel_size=self.B, stride=self.B)
        x_rotated = H(x, mode)
        #cs_ratio_map_rotated = H(cs_ratio_map, mode)
        #saliency_feature = self.E(cs_ratio_map_rotated)
        #x_rotated = x_rotated + self.P(torch.cat([x_rotated, saliency_feature], dim=1))
        x_rotated = x_rotated + self.P(x_rotated)
        return H(x_rotated, mode, inv=True)  # inverse of H

#my_sal
import torchvision.models as models
import torchvision.transforms as transforms
import time
from PIL import Image
import cv2
device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
def get_saliency(secret_batch, base_project_path):

    pre_model = models.mobilenet_v2(pretrained=True).to(device)
    pre_model.eval() 
    transform_for_grad_cam = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    processed_saliency_maps = []
    
    for i in range(secret_batch.size(0)):
        single_secret_tensor = secret_batch[i] 
        original_h, original_w = single_secret_tensor.shape[1], single_secret_tensor.shape[2]

        pil_img = transforms.ToPILImage()(single_secret_tensor.cpu().clamp(0, 1))
        pil_img_rgb = pil_img.convert('RGB')

        img_tensor_for_model = transform_for_grad_cam(pil_img_rgb).unsqueeze(0).to(device)
        
        with torch.enable_grad(): 
            img_tensor_for_model.requires_grad_(True) 
            target_layer = pre_model.features[-1]

            activations = []
            gradients = []

            def hook_forward(module, input, output):
                activations.append(output)

            def hook_backward(module, grad_input, grad_output):
                gradients.append(grad_output[0]) 

            forward_handle = target_layer.register_forward_hook(hook_forward)
            backward_handle = target_layer.register_backward_hook(hook_backward)

            start_grad_cam_time = time.time()
            output = pre_model(img_tensor_for_model)
            
            pred_class = output.argmax(dim=1).item()

            pre_model.zero_grad() 
            output[0, pred_class].backward() 

            forward_handle.remove()
            backward_handle.remove()

        if not activations or not gradients:
            print(f"Warnung: Keine Aktivierungen/Gradienten für Bild {i+1} erfasst.")
            raw_heatmap_np = np.zeros((original_h, original_w), dtype=np.float32)
        else:
            
            weights = torch.mean(gradients[0], dim=[2, 3], keepdim=True)
            heatmap = torch.sum(weights * activations[0], dim=1).squeeze()
            
            heatmap = np.maximum(heatmap.cpu().detach().numpy(), 0)
            if np.max(heatmap) > 0:
                heatmap /= np.max(heatmap)
            else:
                heatmap = np.zeros_like(heatmap)

            raw_heatmap_np = cv2.resize(heatmap, (original_w, original_h))
        
        end_grad_cam_time = time.time()
        #print(f"  Bild {i+1}/{secret_batch.size(0)} Grad-CAM generiert (Dauer: {end_grad_cam_time - start_grad_cam_time:.4f}s)")

        saliency_tensor_1_channel = torch.from_numpy(raw_heatmap_np).float().unsqueeze(0).to(device)
        saliency_tensor_1_channel = torch.clamp(saliency_tensor_1_channel, 0, 1)

        processed_saliency_maps.append(saliency_tensor_1_channel)

    final_saliency_output = torch.stack(processed_saliency_maps, dim=0)
    #print(f"Alle Bilder für Saliency verarbeitet. Finaler Batch Shape für Saliency: {final_saliency_output.shape}")
    saliency_reshaped_for_softmax = final_saliency_output.reshape(
        final_saliency_output.shape[0], 
        final_saliency_output.shape[1], 
        -1 
    )
    
    saliency_softmaxed = F.softmax(saliency_reshaped_for_softmax, dim=2)
    
    saliency_final_processed = saliency_softmaxed.reshape_as(final_saliency_output)

    return saliency_final_processed


# saliency detector
class D(nn.Module):
    def __init__(self, img_nf, base_project_path):
        super(D, self).__init__()
        self.recover = CBR(img_nf, 3, 1, stride=1, groups=1)
        self.model = mobilenet_v3(base_project_path, pretrained=True,)
        self.dupsample = DUpsampling(160, 32, num_class=1)
        
    def forward(self, x):
        return(self.dupsample(self.model.features(self.recover(x)))).reshape(*x.shape[:2], -1).softmax(dim=2).reshape_as(x)
    

def init_net(base_project_path):
    # Instantiate PrepNetwork and move to device
    prep_net = PNet().to(device)
    
    #print("DEBUG: BASE PATH: ", base_project_path)
    checkpoint_path = base_project_path+'output/prep_net_Weights/100_epoch_reg.tar'

    if os.path.isfile(checkpoint_path):
        #print(f"Lade Gewichte von {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Check if the checkpoint is a full model state_dict or a dict containing 'state_dict'
        state_dict_to_load = checkpoint.get('state_dict', checkpoint)

        # Create a new state_dict for PrepNetwork only
        prep_net_state_dict = {}
        for k, v in state_dict_to_load.items():
            # PrepNetwork weights start with 'm1.' in the Netw model
            if k.startswith('m1.'):
                # Remove the 'm1.' prefix
                prep_net_state_dict[k[3:]] = v
        get_saliency
        # Remove 'module.' prefix if it exists (from DataParallel, common in saved checkpoints)
        filtered_prep_net_state_dict = OrderedDict()
        for k, v in prep_net_state_dict.items():
            name = k[7:] if k.startswith('module.') else k
            filtered_prep_net_state_dict[name] = v

        try:
            prep_net.load_state_dict(filtered_prep_net_state_dict)
            #print("PrepNetwork Gewichte erfolgreich geladen.")
        except RuntimeError as e:
            print(f"Fehler beim Laden der PrepNetwork Gewichte: {e}")
            print("Stelle sicher, dass der Checkpoint die korrekten PrepNetwork Gewichte enthält und die Architektur übereinstimmt.")
    else:
        print(f"Warnung: Kein Checkpoint unter '{checkpoint_path}' gefunden. PrepNetwork wird mit zufälligen Gewichten ausgeführt.")

    prep_net.eval() # Set to evaluation modeget_saliency

    return prep_net

def get_prep(img_batch, prep_net):

    with torch.no_grad():
        feature_maps = prep_net(img_batch)
    #feature_maps = feature_maps.squeeze(0)
    mean_feature_map_np = torch.mean(feature_maps, dim=1, keepdim=True)
    return mean_feature_map_np

# error correction of BRA
def batch_correct(Q, target_sum, N):
    b, l = Q.shape
    i, max_desc_step = 0, 10
    while True:
        i += 1
        Q = torch.clamp(Q, 0, N).round()
        d = Q.sum(dim=1) - target_sum  # batch delta
        if float(d.abs().sum()) == 0.0:
            break
        elif i < max_desc_step:  # 1: uniform descent
            Q = Q - (d / l).reshape(-1, 1).expand_as(Q)
        else:  # 2: random allocation
            for j in range(b):
                D = np.random.multinomial(int(d[j].abs().ceil()), [1.0 / l] * l, size=1)
                Q[j] -= int(d[j].sign()) * torch.Tensor(D).squeeze(0).to(Q.device)
    return Q

def stretch_val(S):
    TARGET_S_MIN = 0.00009
    TARGET_S_MAX = 0.0003
    # Values < 1.0 will compress the stretch (e.g., 0.5 for half intensity).
    # Values > 1.0 will expand the stretch.
    SALIENCY_STRETCH_FACTOR = 0.5

    S_min = S.min()
    S_max = S.max()
    if S_max > S_min: # Avoid division by zero if S is constant (all values are the same)
        S_normalized_0_1 = (S - S_min) / (S_max - S_min) 
        effective_target_range_diff = (TARGET_S_MAX - TARGET_S_MIN) * SALIENCY_STRETCH_FACTOR
        S_stretched = S_normalized_0_1 * effective_target_range_diff + TARGET_S_MIN

    else:
        # If S is constant (no variation), set it to the midpoint of the target range
        S_stretched = torch.ones_like(S) * ((TARGET_S_MIN + TARGET_S_MAX) / 2.0)
    S = torch.clamp(S_stretched, min=TARGET_S_MIN, max=TARGET_S_MAX)
    return S
    
class Adaptive_CS(nn.Module):
    def __init__(self, project_path, sal_for_cs, color, phase_num, B, img_nf, Phi_init):
        super(Adaptive_CS, self).__init__()
        self.phase_num = phase_num
        self.phase_num_minus_1 = phase_num - 1
        self.B = B
        self.color = color
        self.sal_for_cs = sal_for_cs
        self.N = B * B
        self.base_project_path = project_path
        self.Phi = nn.Parameter(Phi_init.reshape(self.N, self.N))
        self.RS = nn.ModuleList([Phase(img_nf, B) for _ in range(phase_num)])
        self.D = D(img_nf, self.base_project_path)
        self.index_mask = torch.arange(1, self.N + 1)

    def forward(self, x, q, h_sr, l_sr, share = None, gamma=0.2822):
        b, c, h, w = x.shape
        if self.color:
            x_yuv = kcolor.rgb_to_yuv(x)
            x_y = x_yuv[:, 0:1, :, :]   # Y-channel (Luminanz, greyscale)
            x_uv = x_yuv[:, 1:3, :, :]  # UV-channels (Chrominanz, color info)
            form = self.N
            #x_uv_downsampled = F.interpolate(x_uv, size=(h//2, w//2), mode='bicubic', align_corners=False)
        else:
            x_y = x
            x_uv = 0
            form = c * self.N

        #print(f"DEBUG: Client x shape before unfold: {x.shape}")
        x_unfold = F.unfold(x_y, kernel_size=self.B, stride=self.B).permute(0, 2, 1) 
        #print(f"DEBUG: Client x shape after unfold: {x_unfold.shape}")
        l = x_unfold.shape[1]  
        block_stack = x_unfold.reshape(-1, form, 1)  
        L = block_stack.shape[0]  
        Phi_stack = self.Phi.unsqueeze(0).repeat(L, 1, 1)
        index_mask = self.index_mask.unsqueeze(0).repeat(L, 1).to(Phi_stack.device)
        
        q = int(np.ceil(0.5 * self.B * self.B))
        q_basic = int(np.ceil(gamma * q))
        #print(f"DEBUG: q: {q}")

        Phi_ori = Phi_stack.clone()
        #print(f"DEBUG: Phi_ori shape: {Phi_ori.shape}")
        PhiT_Phi_ori = Phi_ori.permute(0, 2, 1).matmul(Phi_ori)
        #print(f"DEBUG: PhiT_Phi_ori shape: {PhiT_Phi_ori.shape} and block_Stack_shape: {block_stack.shape}")
        PhiT_y_ori = PhiT_Phi_ori.matmul(block_stack).reshape(b, l, -1).permute(0, 2, 1)
        #print(f"DEBUG: PhiT_y_ori shape: {PhiT_y_ori.shape}")
        PhiT_y_ori = F.fold(PhiT_y_ori, output_size=(h, w), kernel_size=self.B, stride=self.B)
        #print(f"DEBUG: PhiT_y_ori shape: {PhiT_y_ori.shape}")
        if share is not None:
            S = share
            #print(f"DEBUG: S shape1: {S.shape}")
            if not self.sal_for_cs:
                S = stretch_val(S)
                #print(f"DEBUG: S shape2: {S.shape}")
        else:
            # 2. adaptive CS ratio allocation
            if not self.sal_for_cs:
                #print(f"DEBUG: PhiT_y_ori shape: {PhiT_y_ori.shape}")
                PhiT_y_ori_3_channel = PhiT_y_ori.repeat(1, 3, 1, 1)
                #print(f"DEBUG: PhiT_y_ori3 shape: {PhiT_y_ori_3_channel.shape}")
                S = get_prep(PhiT_y_ori_3_channel, init_net(self.base_project_path))
                S = stretch_val(S)
                #print(f"DEBUG: S shape3: {S.shape}")
            else:
                S = get_saliency(PhiT_y_ori, self.base_project_path)
                #print(f"DEBUG: S shape4: {S.shape}")
                #print(f"DEBUG: Saliency map shape: {S.shape}")
        
        Q = (q - q_basic) * l * S 

        #print(f"DEBUG: Q shape: {Q.shape}")
        Q_unfold_before_threshold = F.unfold(Q, kernel_size=self.B, stride=self.B).permute(0, 2, 1).sum(dim=2)  

        Q_unfold = batch_correct(Q_unfold_before_threshold, (q - q_basic) * l, self.N - q_basic) + q_basic 

        fea_val = self.B * self.B
        thre = math.ceil(fea_val*0.5)

        if self.sal_for_cs:
            Q_unfold = torch.where(Q_unfold > thre, math.ceil(fea_val * l_sr), math.ceil(fea_val * h_sr))
        else:
            Q_unfold = torch.where(Q_unfold > thre, math.ceil(fea_val * h_sr), math.ceil(fea_val * l_sr))
             

        Q_mid1 = Q_unfold
        Q_mid2 = Q_mid1.cpu().numpy()
        q_mid = [0]*Q_unfold.numel()
        for i in range(Q_unfold.numel()):
            q_mid[i] = Q_mid2[0][i]

        list0 = 0
        for i in range(Q_unfold.numel()):
            list1 = [index_mask[i] > q_mid[i]]         
            if list0 == 0:
                list0 = list1
            else:
                list0.extend(list1)

        final_list =torch.stack(list0,0)
        
        # 3. content-aware sampling
        Phi_basic = Phi_stack.clone()
        Phi_basic[final_list] = 0
        PhiT_Phi_basic = Phi_basic.permute(0, 2, 1).matmul(Phi_basic)
        PhiT_y = PhiT_Phi_basic.matmul(block_stack)
        PhiT_y_basic = PhiT_y.reshape(b, l, -1).permute(0, 2, 1)
        PhiT_y_basic = F.fold(PhiT_y_basic, output_size=(h, w), kernel_size=self.B, stride=self.B)
        shape_info = [b, l, h, w]


        #print(f"DEBUG: PhiT_Phi_basic shape: {PhiT_Phi_basic.shape}")

        return PhiT_Phi_basic, PhiT_y_basic, x_uv, L, shape_info
    
