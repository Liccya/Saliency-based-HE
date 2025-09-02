import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from utils import H
import math
import kornia.color as kcolor
from mobilenet_v3 import mobilenet_v3


feature_map = False

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


# saliency detector
class D(nn.Module):
    def __init__(self, img_nf, base_project_path):
        super(D, self).__init__()
        self.recover = CBR(img_nf, 3, 1, stride=1, groups=1)
        self.model = mobilenet_v3(base_project_path, pretrained=True)
        self.dupsample = DUpsampling(160, 32, num_class=1)
        
    def forward(self, x):
        return(self.dupsample(self.model.features(self.recover(x)))).reshape(*x.shape[:2], -1).softmax(dim=2).reshape_as(x)

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
    
class Adaptive_CS(nn.Module):
    def __init__(self, project_path, color, phase_num, B, img_nf, Phi_init):
        super(Adaptive_CS, self).__init__()
        self.phase_num = phase_num
        self.phase_num_minus_1 = phase_num - 1
        self.B = B
        self.N = B * B
        self.color = color
        self.base_project_path = project_path
        self.Phi = nn.Parameter(Phi_init.reshape(self.N, self.N))
        self.RS = nn.ModuleList([Phase(img_nf, B) for _ in range(phase_num)])
        self.D = D(img_nf, self.base_project_path)
        self.index_mask = torch.arange(1, self.N + 1)
        

    def forward(self, modes, input):

        PhiT_Phi_basic, PhiT_y_basic, x_uv_downsampled, L, shape_info = input

        b, l, h, w = shape_info
        PhiT_Phi = 0
        PhiT_y = 0
        
        PhiT_Phi += PhiT_Phi_basic
        PhiT_y += F.unfold(PhiT_y_basic, kernel_size=self.B, stride=self.B).permute(0, 2, 1).reshape(L, -1, 1)
        x_y_reconstructed = PhiT_y
        
        for i in range(self.phase_num): 
            x_y_reconstructed = self.RS[i](x_y_reconstructed, PhiT_Phi, PhiT_y, modes[i], shape_info)
            if i < self.phase_num_minus_1:
                x_y_reconstructed = F.unfold(x_y_reconstructed, kernel_size=self.B, stride=self.B).permute(0, 2, 1)
                x_y_reconstructed = x_y_reconstructed.reshape(L, -1, 1)
        
        if self.color:
            # Skaliere die UV-Kanäle wieder auf die ursprüngliche Größe hoch
            x_uv_upsampled =x_uv_downsampled#= F.interpolate(x_uv_downsampled, size=(h, w), mode='bicubic', align_corners=False)

            # Füge den rekonstruierten Y-Kanal und die hochskalierten UV-Kanäle zusammen
            x_yuv_reconstructed = torch.cat([x_y_reconstructed, x_uv_upsampled], dim=1)
            
            # Konvertiere zurück in RGB
            x = kcolor.yuv_to_rgb(x_yuv_reconstructed)
        else:
            x = x_y_reconstructed
        
        return x
