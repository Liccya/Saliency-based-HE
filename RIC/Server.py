
from distutils.log import error
import os
import sys
os.environ['CUDA_VISIBLE_DEVICES'] = "0" 
import numpy as np
from skimage.metrics import structural_similarity, peak_signal_noise_ratio
import torch
from torch.autograd import Variable
from collections import OrderedDict
from random import shuffle
from torchvision.utils import make_grid,save_image
import socket
import torch.optim as optim
import juncw
import argparse
import random
import io, cv2
from Server_net import Net
from Train_Net import Net as TrainNet
from pytorch_msssim import ssim, ms_ssim
import PerceptualSimilarity.models
from IoT_small import Netw as Netw_small
from IoT_regular import Netw as Netw_regular
from pre_process_image import process_image, init_model, generate_grad_cam_overlay 
import torchvision.transforms as transforms
from PIL import Image 
from decompress_model import Adaptive_CS as Adaptive_CS_decompress
from IoT_compress import Adaptive_CS as Adaptive_CS_compress

playground_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'pytorch-playground-master'))
sys.path.append(playground_path)
from utee import selector


def valmetric(S_prime, mix_img, S,C,  B,labels):
    if not color:
        S_prime = S_prime.repeat(1,3,1,1)
    ''' Calculates loss'''
    outputs = target_model(mix_img)
    pre = torch.argmax(outputs,dim=1)
    acc_num = len(torch.where(pre==labels)[0])    
    psnr, ssim, mse_s,mse_c,mean_pixel_error,lpips_error = [], [], [],[],[],[]
    norm_S =  convert1(S*_std_torch+_mean_torch)
    norm_S_prime = convert1(S_prime)
    norm_mix_image = convert1(mix_img)
    mse_sc,psnr_sc,ssim_sc,lpips_sc = [], [], [],[]
    for i in range(len(S_prime)):
        mse_s.append(float(torch.norm((S*_std_torch+_mean_torch)[i]-S_prime[i])))
        mse_c.append(float(torch.norm(C[i]-mix_img[i])))
        psnr.append(peak_signal_noise_ratio(norm_S[i],norm_S_prime[i],data_range=255))
        ssim.append(structural_similarity(norm_S[i],norm_S_prime[i],win_size=11, data_range=255.0, channel_axis=-1))
        mean_pixel_error.append(float(torch.sum(torch.abs(torch.round((S*_std_torch+_mean_torch)[i]*255)-torch.round(S_prime[i]*255)))/(3*imgsize*imgsize)))
        tmp = modellp.forward((S*_std_torch+_mean_torch)[i], S_prime[i],normalize=True)
        lpips_error.append(float(tmp))
        #mix_image and secret image
        mse_sc.append(float(torch.norm((S*_std_torch+_mean_torch)[i]-mix_img[i])))
        psnr_sc.append(peak_signal_noise_ratio(norm_S[i],norm_mix_image[i],data_range=255))
        ssim_sc.append(structural_similarity(norm_S[i],norm_mix_image[i],win_size=11, data_range=255.0, channel_axis=-1))
        tmp = modellp.forward((S*_std_torch+_mean_torch)[i], mix_img[i],normalize=True)
        lpips_sc.append(float(tmp))
    return acc_num, np.sum(mse_s), np.sum(mse_c),np.sum(psnr),np.sum(ssim),np.sum(mean_pixel_error),np.sum(lpips_error),np.sum(lpips_sc),np.sum(mse_sc),np.sum(psnr_sc),np.sum(ssim_sc)


def getAdvZ(img_size,labels,batch_size,minibs):
    '''creates cover image depending on specified target model'''
    c=10
    k=5
    st = 100
    z0 = torch.rand((minibs,3,img_size,img_size))    
    cwatt = juncw.CW(target_model,c=c,kappa=k,steps=st,targeted=True,target_labels=labels)  
   
    succ = False
    adv = cwatt(z0,labels)
    del cwatt
    outputs = target_model(adv)
    _,pre = torch.max(outputs,1)

    succ = len(torch.where(pre==labels)[0])

    if not color:
        adv_grayscale_list = []
        # Iteriere durch den Batch, um jedes Bild einzeln zu konvertieren
        for i in range(adv.shape[0]):
            # PyTorch Tensor (C, H, W) zu NumPy Array (H, W, C) und auf 0-255 skalieren
            adv_np_rgb_255 = (adv[i].permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)
            
            # Konvertiere RGB zu BGR für OpenCV, dann zu YCrCb und nimm den Y-Kanal
            adv_np_bgr_255 = cv2.cvtColor(adv_np_rgb_255, cv2.COLOR_RGB2BGR)
            adv_np_ycrcb = cv2.cvtColor(adv_np_bgr_255, cv2.COLOR_BGR2YCrCb)
            adv_grayscale_np = adv_np_ycrcb[:, :, 0] # Y-Kanal extrahieren
            
            # Konvertiere zurück zu PyTorch Tensor und normalisiere zu 0-1
            # Unsqueeze(0) fügt die Kanal-Dimension (1) hinzu: (1, H, W)
            adv_grayscale_tensor = torch.from_numpy(adv_grayscale_np).unsqueeze(0).float() / 255.0
            adv_grayscale_list.append(adv_grayscale_tensor)
        
        # Stapel die einzelnen Graustufen-Tensoren zu einem Batch-Tensor (minibs, 1, img_size, img_size)
        adv = torch.stack(adv_grayscale_list, dim=0).to(device)

    return adv,succ

def convert1(img):
    img = img * 255.0
    img = img.permute(0, 2, 3, 1).cpu().detach().numpy()
    return img

def load_resume_checkpoint(path, model, target_device):

    filepath = os.path.join(path, 'reveal_network_weights.pth')
    if os.path.isfile(filepath):
        print(f"=> Loading resume checkpoint '{filepath}'")
        checkpoint = torch.load(filepath, map_location='cpu')
        checkpoint_state_dict = checkpoint.get('state_dict', checkpoint)
        filtered_model_state_dict = OrderedDict()
        for k, v in checkpoint_state_dict.items():
            name = k#[7:] if k.startswith('module.') else k
            if name.startswith('module.'):
                name = name[len('module.'):]
            name = f"{name}"
            filtered_model_state_dict[name] = v
        try:
            model.m3.load_state_dict(filtered_model_state_dict, strict=True)
        except RuntimeError as e:
            print(f"  WARNING: Error loading model state_dict strictly: {e}")
            print("  Attempting to load with strict=False (some keys might be missing/mismatched).")
            model.load_state_dict(filtered_model_state_dict, strict=False)

        epoch = checkpoint.get('epoch', 0)
        model.to(target_device)
        print(f"=> Resumed from epoch {epoch} and model moved to {target_device}.")

        return model, epoch

    else:
        print(f"=> No checkpoint found at '{filepath}'. Nothing loaded.")
        return model,  0

    
def load_resume_checkpoint_old(filepath, model, target_device):
    if os.path.isfile(filepath):
        print(f"=> Loading resume checkpoint '{filepath}'")
        checkpoint = torch.load(filepath, map_location='cpu')
        checkpoint_state_dict = checkpoint.get('state_dict', checkpoint)
        filtered_model_state_dict = OrderedDict()

        # Print all keys found in the checkpoint file
        print("Keys found in the checkpoint file:")
        for k in checkpoint_state_dict.keys():
            print(f"  - {k}")
        print("-" * 50)
        

        for k, v in checkpoint_state_dict.items():
            name = k#[7:] if k.startswith('module.') else k
            if name.startswith('module.'):
                name = name[len('module.'):]
            name = f"m3.{name}"
            filtered_model_state_dict[name] = v
        try:
            model.load_state_dict(filtered_model_state_dict, strict=True)
        except RuntimeError as e:
            print(f"  WARNING: Error loading model state_dict strictly: {e}")
            print("  Attempting to load with strict=False (some keys might be missing/mismatched).")
            model.load_state_dict(filtered_model_state_dict, strict=False)

        epoch = checkpoint.get('epoch', 0)
        model.to(target_device)
        print(f"=> Resumed from epoch {epoch} and model moved to {target_device}.")
        return model, epoch

    else:
        print(f"=> No checkpoint found at '{filepath}'. Nothing loaded.")
        return model,  0
    
def split_checkpoint_by_modules(full_checkpoint_path, output_dir, module_prefix_map):
    
    if not os.path.exists(full_checkpoint_path):
        print(f"Fehler: Checkpoint not found at '{full_checkpoint_path}'")
        return

    os.makedirs(output_dir, exist_ok=True)
    print(f"Load checkpoint from '{full_checkpoint_path}'...")
    checkpoint = torch.load(full_checkpoint_path, map_location='cpu')
    full_state_dict = checkpoint.get('state_dict', checkpoint)
    
    for module_name_friendly, prefix in module_prefix_map.items():
        sub_module_state_dict = OrderedDict()
        
        for k, v in full_state_dict.items():
            key_without_dp_prefix = k[7:] if k.startswith('module.') else k
            if key_without_dp_prefix.startswith(prefix):
                filtered_key = key_without_dp_prefix[len(prefix):]
                sub_module_state_dict[filtered_key] = v
                
        if sub_module_state_dict:
            output_filepath = os.path.join(output_dir, f"{module_name_friendly}_weights.pth")
            torch.save(sub_module_state_dict, output_filepath)
            print(f"  Saved '{module_name_friendly}' weights in '{output_filepath}' ({len(sub_module_state_dict)} key)")

        else:
            print(f"  WARNING: No key found for submodule '{module_name_friendly}' with prefix '{prefix}'.")

def attack1_loss(S_prime, mix_img, S,C,  B,labels):
    
    ''' Calculates loss specified on the paper.'''
    #print("S_prime range:", S_prime.min().item(), S_prime.max().item())
    #print("S range:", S.min().item(), S.max().item())
    if not color:
        mix_img_3chan = mix_img.repeat(1, 3, 1, 1)
    else:
        mix_img_3chan = mix_img

    loss_secret = torch.nn.functional.mse_loss(S_prime,  S*_std_torch+_mean_torch)
    loss_cover= torch.nn.functional.mse_loss(mix_img,  C)
    outputs = target_model(mix_img_3chan)
    classloss = torch.mean(getCEloss(labels,outputs))
    ssim_loss = ssim(S,mix_img,data_range=1.0, size_average=True)
    
    loss_all =   B*loss_secret  + classloss
    
    return loss_all, loss_secret,classloss,loss_cover, ssim_loss

def getCEloss(labels,outputs):
    one_hot_labels = torch.eye(len(outputs[0]), device=labels.device)[labels]

    i, _ = torch.max((1-one_hot_labels)*outputs, dim=1) # get the second largest logit
    j = torch.masked_select(outputs, one_hot_labels.bool()) # get the largest logit

    return torch.clamp((i-j), min=-kappa)

def save_checkpoint(state, filename):
    checkpointname = filename+'_checkpoint.pth.tar'
    os.makedirs(os.path.dirname(checkpointname), exist_ok=True)
    torch.save(state, checkpointname)

def train_old_old(train_loader, beta, learning_rate, batch_size, start_epoch=0, num_epochs = 200):
    
    output_dir = os.path.join(outputname, 'model_train_epochs')
    print(output_dir)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    net_train = TrainNet(base_project_path, True, True,0.8, small, compress, encrypt, color)
    net_train.to(device)
    optimizer = optim.Adam(net_train.parameters(), lr=learning_rate)

    for epoch in range(start_epoch, num_epochs):
        print(f"Starting Epoch {epoch+1}/{num_epochs}")
        net_train.train() 
        train_losses = []
        for idx, train_batch in enumerate(train_loader):
            net_train.train() 

            train_secrets_original, labels  = train_batch 
            train_secrets_original = train_secrets_original.to(device)
            
            labels = labels.to(device)
            train_covers,succ = getAdvZ(imgsize,labels,batch_size,len(train_secrets_original))

            optimizer.zero_grad()

            if not color:
                ###change to grey scale###
                train_secrets_np = train_secrets_original.permute(0, 2, 3, 1).cpu().numpy() * 255.0
                train_secrets_gs_list = []
                for i in range(train_secrets_np.shape[0]):
                    train_secrets_bgr = cv2.cvtColor(train_secrets_np[i].astype(np.float32), cv2.COLOR_RGB2BGR)
                    gray_channel = cv2.cvtColor(train_secrets_bgr, cv2.COLOR_BGR2YCrCb)[:, :, 0]
                    train_secrets_gs_list.append(gray_channel) 
                train_secrets_gs = torch.from_numpy(np.stack(train_secrets_gs_list)).unsqueeze(1).to(device).float() / 255.0 
                ###change to greyscale end ###
                train_secrets_original = train_secrets_gs
            
            mix_img, recover_secret = net_train(train_secrets_original, train_covers, train=True)
            
            train_loss, train_loss_secret, attloss, loss_cover, ssim_loss = attack1_loss(recover_secret,mix_img, train_secrets_original,train_covers,beta,labels)

            
            train_loss.backward()
            optimizer.step()
            
            train_losses.append(train_loss.item())

            # Prints mini-batch losses - use if needed
            '''if idx%100==0:
            print('Training: Batch {0}/{1}. Loss of {2:.4f},secret loss of {3:.4f}, attack1_loss of {4:.4f}, loss_cover {5:.5f}, ssim_loss: {6:.4f}'.format(idx+1, len(train_loader), train_loss.data,  train_loss_secret.data,attloss.data,loss_cover,ssim_loss))'''

        modelsavepath = os.path.join(output_dir,'Epoch_{}'.format(epoch))
        save_checkpoint({
            'epoch': epoch + 1, # Save the epoch *after* this one completes
            'state_dict': net_train.state_dict(), 
            'optimizer' : optimizer.state_dict()}, modelsavepath)
                
        mean_train_loss = np.mean(train_losses)
    
        print ('Epoch [{0}/{1}], Average_loss: {2:.4f}'.format(
                epoch+1, num_epochs, mean_train_loss))
    
    last_checkpoint_path = os.path.join(output_dir,'last')
    save_checkpoint({
        'epoch': epoch + 1,
        'state_dict': net_train.state_dict(),
        'optimizer': optimizer.state_dict(),
        }, last_checkpoint_path)

    module_mapping = {
        'prep_network': 'm1.',
        'hide_network': 'm2.',
        'reveal_network': 'm3.'
    }
    split_checkpoint_by_modules(last_checkpoint_path+'_checkpoint.pth.tar', MODELS_PATH, module_mapping)

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


def train(train_loader, beta, learning_rate, batch_size, start_epoch=0, num_epochs = 200):
    
    output_dir = os.path.join(outputname, 'model_train_epochs')
    print(output_dir)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    
    color_nets = color

    net_rev = Net(base_project_path, residual_en, residual_de, lambda_net, small, compress, encrypt, color_nets).to(device)
    if small:
        net_hide = Netw_small(color_nets, residual_en,residual_de,lambda_net).to(device)
    else:
        net_hide = Netw_regular(color_nets, residual_en,residual_de,lambda_net).to(device)
    
    if compress:
        epoch = 320
        N_p = 13
        B = 8
        N = B*B
        model_compress = Adaptive_CS_compress(base_project_path, sal_cs, color, N_p, B, 1, torch.zeros(N, N))
        model_decompress = Adaptive_CS_decompress(base_project_path, N_p, B, 1, torch.zeros(N, N))
        model_compress = torch.nn.DataParallel(model_compress).to(device)
        model_decompress = torch.nn.DataParallel(model_decompress).to(device)
        model_dir = '%s/layer_%d_block_%d' % (os.path.join(base_project_path,'compress_Data/model'), N_p, B)
        model_compress.load_state_dict(torch.load('%s/net_params_%d.pkl' % (model_dir, 320), map_location=device))
        model_decompress.load_state_dict(torch.load('%s/net_params_%d.pkl' % (model_dir, 320), map_location=device))


    net_hide.to(device)
    net_rev.to(device)
    all_parameters = list(net_hide.parameters()) + list(net_rev.parameters())
    optimizer = optim.Adam(all_parameters, lr=learning_rate)

    for epoch in range(start_epoch, num_epochs):
        print(f"Starting Epoch {epoch+1}/{num_epochs}")
        train_losses = []
        for idx, train_batch in enumerate(train_loader):
            net_rev.train()
            net_hide.train()  
            train_secrets_original, labels  = train_batch 
            train_secrets_original = train_secrets_original.to(device)

            secret_sal=get_saliency(train_secrets_original)
            
            labels = labels.to(device)
            train_covers,succ = getAdvZ(imgsize,labels,batch_size,len(train_secrets_original))

            optimizer.zero_grad()

            if not color:
                ###change to grey scale###
                train_secrets_np = train_secrets_original.permute(0, 2, 3, 1).cpu().numpy() * 255.0
                train_secrets_gs_list = []
                for i in range(train_secrets_np.shape[0]):
                    train_secrets_bgr = cv2.cvtColor(train_secrets_np[i].astype(np.float32), cv2.COLOR_RGB2BGR)
                    gray_channel = cv2.cvtColor(train_secrets_bgr, cv2.COLOR_BGR2YCrCb)[:, :, 0]
                    train_secrets_gs_list.append(gray_channel) 
                train_secrets_gs = torch.from_numpy(np.stack(train_secrets_gs_list)).unsqueeze(1).to(device).float() / 255.0 
                ###change to greyscale end ###
                train_secrets_original = train_secrets_gs
            
            mix_img = net_hide(train_secrets_original, secret_sal, train_covers, train=True)
            if compress:
                x_2_compressed=[]
                x_2 = mix_img
                for i in range(x_2.shape[0]):
                    x_2_single = x_2[i:i+1]
                    PhiT_Phi_basic, PhiT_y_basic, x_uv_downsampled, L, shape_info = model_compress(
                    x_2_single, 
                    int(np.ceil(0.01 * N)),
                    0.5,
                    0.01,
                    )   
                    x_2_single_compressed=[PhiT_Phi_basic, PhiT_y_basic, x_uv_downsampled, L,shape_info]
                    x_2_compressed.append(x_2_single_compressed)
                    #decompress in training script so that backtracking affects this
                    reconstructed_images = []
                    rand_modes = [random.randint(0, 7) for _ in range(N_p)]
                    for i in x_2_compressed:
                        x_2_single = model_decompress(
                                rand_modes,
                                i
                            )
                        reconstructed_images.append(x_2_single)
                    x_2= torch.cat(reconstructed_images, dim=0)
                mix_img = x_2
            #print(f"DEBUG: Client x_2 0 after compress: {x_2[0]}")

            _, recover_secret = net_rev(mix_img, train_covers, labels, None, train=True)
            
            train_loss, train_loss_secret, attloss, loss_cover, ssim_loss = attack1_loss(recover_secret,mix_img, train_secrets_original,train_covers,beta,labels)

            
            train_loss.backward()
            optimizer.step()
            
            train_losses.append(train_loss.item())

            # Prints mini-batch losses - use if needed
            '''if idx%100==0:
            print('Training: Batch {0}/{1}. Loss of {2:.4f},secret loss of {3:.4f}, attack1_loss of {4:.4f}, loss_cover {5:.5f}, ssim_loss: {6:.4f}'.format(idx+1, len(train_loader), train_loss.data,  train_loss_secret.data,attloss.data,loss_cover,ssim_loss))'''

        modelsavepath = os.path.join(output_dir,'Epoch_{}_hide'.format(epoch))
        save_checkpoint({
            'epoch': epoch + 1, # Save the epoch *after* this one completes
            'state_dict': net_hide.state_dict(), 
            'optimizer' : optimizer.state_dict()}, modelsavepath)
                
        modelsavepath = os.path.join(output_dir,'Epoch_{}_rev'.format(epoch))
        save_checkpoint({
            'epoch': epoch + 1, # Save the epoch *after* this one completes
            'state_dict': net_rev.state_dict(), 
            'optimizer' : optimizer.state_dict()}, modelsavepath)
                        
        mean_train_loss = np.mean(train_losses)
    
        print ('Epoch [{0}/{1}], Average_loss: {2:.4f}'.format(
                epoch+1, num_epochs, mean_train_loss))
    
    last_checkpoint_path = os.path.join(output_dir,'last_hide')
    save_checkpoint({
        'epoch': epoch + 1,
        'state_dict': net_hide.state_dict(),
        'optimizer': optimizer.state_dict(),
        }, last_checkpoint_path)

    last_checkpoint = os.path.join(outputname,'model/reveal_network_weights')
    save_checkpoint({
        'epoch': epoch + 1,
        'state_dict': net_rev.state_dict(),
        'optimizer': optimizer.state_dict(),
        }, last_checkpoint)

    module_mapping = {
        'prep_network': 'm1.',
        'hide_network': 'm2.',
        'reveal_network': 'm3.'
    }
    split_checkpoint_by_modules(last_checkpoint_path+'_checkpoint.pth.tar', MODELS_PATH, module_mapping)

def send_tensor(sock, tensor):
    buffer = io.BytesIO()
    torch.save(tensor, buffer)
    sock.sendall(len(buffer.getvalue()).to_bytes(4, 'big'))
    sock.sendall(buffer.getvalue())

def test():
    
    total = 0
    acctotal = 0
    l2loss_secrets = 0 
    l2loss_covers = 0 
    psnr_secrets =0 
    ssim_secrets =0
    mean_pixel_errors =0 
    total_all = 0
    lpips_errors,lpips_scs,mse_scs,psnr_scs,ssim_scs = 0.,0.,0.,0.,0.
    client_iteration_times = []
    current_mem = []
    peak_mem = []
    saved_data = []

    
    HOST = '0.0.0.0'
    PORT = 5001
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind((HOST, PORT))
    server_socket.listen(1)
    print("Connecting...")

    conn, addr = server_socket.accept()
    print(f"Connected to {addr}")

    send_tensor(conn, [small, compress, encrypt, sal_cs, share_guide, outputname, base_project_path])

    
    os.makedirs('{}/testimages/.png'.format(outputname), exist_ok=True)

    residual_en = True
    residual_de = True
    net = Net(base_project_path, residual_en, residual_de, lambda_net, compress, encrypt).to(device)
    net, start_epoch = load_resume_checkpoint(MODELS_PATH, net, device)


    for idx, train_batch in enumerate(test_loader):

        if idx > 2:
            break

        train_secrets, labels  = train_batch
        train_secrets = train_secrets.to(device)
        total_all += train_secrets.shape[0]
        #if total_all<=1000:
        #    continue
        #if total_all>1500:
        #    break
        total  += train_secrets.shape[0]

        labels = labels.to(device)

        train_covers,succ = getAdvZ(imgsize,labels,batch_size,len(train_secrets))
       
        train_secrets = Variable(train_secrets, requires_grad=False)
        train_covers = Variable(train_covers, requires_grad=False)

        train_secrets = train_secrets.to(device)
        train_covers = train_covers.to(device)
        net.eval()

        if not color:
            ###change to grey scale###
            #print(f"DEBUG: Secret shape before greyscale: {train_secrets.shape}")
            train_secrets_np = train_secrets.permute(0, 2, 3, 1).cpu().numpy() * 255.0
            train_secrets_gs_list = []
            for i in range(train_secrets_np.shape[0]):
                train_secrets_bgr = cv2.cvtColor(train_secrets_np[i].astype(np.float32), cv2.COLOR_RGB2BGR)
                gray_channel = cv2.cvtColor(train_secrets_bgr, cv2.COLOR_BGR2YCrCb)[:, :, 0]
                train_secrets_gs_list.append(gray_channel) 
            train_secrets_gs = torch.from_numpy(np.stack(train_secrets_gs_list)).unsqueeze(1).to(device).float() / 255.0 
            #print(f"DEBUG: Secret shape after greyscale: {train_secrets_gs.shape}")
            ###change to greyscale end ###
            train_secrets = train_secrets_gs

        mix_img, recover_secret, duration, mem, saved_mb = net(train_secrets, train_covers, labels, conn)

        print(f"Server: Data saved through compression {idx}: {saved_mb:.4f} MB")
        saved_data.append(saved_mb)

        if duration is not None:
            client_iteration_times.append(duration)
            print(f"Server: Client-Iterationtime for Batch {idx}: {duration:.4f}s")
        else:
            print("Server: No Iteration time received or client closed.")
            break 
        if mem is not None:
            current_mem.append(mem[0]/ 10**6)
            peak_mem.append(mem[1]/ 10**6)
            print(f"Server: Client-Memory-measure {idx}: {mem[0]/ 10**6:.4f} MB")
            print(f"Server: Client-Peak-Memory-measure {idx}: {mem[1]/ 10**6:.4f} MB")
        else:
            print("Server: No Memory measurements received or client closed.")
            break 
        
        if not color:
            mix_img = mix_img.repeat(1, 3, 1, 1)
            train_covers = train_covers.repeat(1, 3, 1, 1)

        acc_num, l2loss_secret,l2loss_cover,psnr_secret,ssim_secret,mean_pixel_error,lpips_error,lpips_sc,mse_sc,psnr_sc,ssim_sc= \
            valmetric(recover_secret,mix_img, train_secrets,train_covers,beta,labels)
        acctotal += int(acc_num)
        l2loss_secrets += float(l2loss_secret)
        l2loss_covers += float(l2loss_cover)
        psnr_secrets += float(psnr_secret)
        ssim_secrets += float(ssim_secret)
        mean_pixel_errors += float(mean_pixel_error)
        lpips_errors += float(lpips_error)
        lpips_scs += float(lpips_sc)
        mse_scs += float(mse_sc)
        psnr_scs += float(psnr_sc)
        ssim_scs += float(ssim_sc)

        print("classification success rate:{}/{}={:.6f}".format(acctotal,total,acctotal/total))
        print("avg. l2loss_secrets:{:.6f}".format(l2loss_secrets/total))
        print("avg. l2loss_covers:{:.6f}".format(l2loss_covers/total))
        print("avg. psnr_secrets:{:.6f}".format(psnr_secrets/total))
        print("avg. ssim_secrets:{:.6f}".format(ssim_secrets/total))
        print("avg. mean_pixel_errors:{:.6f}".format(mean_pixel_errors/total))
        print("avg. lpips_errors:{:.6f}".format(lpips_errors/total))
        print("avg. lpips_scs:{:.6f}".format(lpips_scs/total))
        print("avg. mse_scs:{:.6f}".format(mse_scs/total))
        print("avg. psnr_scs:{:.6f}".format(psnr_scs/total))
        print("avg. ssim_scs:{:.6f}".format(ssim_scs/total))
        print("avg. duration on client side: {:.6f}s".format(sum(client_iteration_times) / len(client_iteration_times)))
        print("avg. memory usage on client side: {:.6f}MB".format(sum(current_mem) / len(current_mem)))
        print("avg. peak memory usage on client side: {:.6f}MB".format(sum(peak_mem) / len(peak_mem)))
        print("avg. data saved through compression: {:.6f}MB".format(sum(saved_data) / len(saved_data)))
        print("")

        recover_vis = torch.clamp(recover_secret[:4] * _std_torch + _mean_torch, 0, 1)
        #print(f"DEBUG: train_secrets shape: {train_secrets.shape}")
        #print(f"DEBUG: train_covers shape: {train_covers.shape}")
        #print(f"DEBUG: mix_img shape: {mix_img.shape}")
        #print(f"DEBUG: recover_vis shape: {recover_vis.shape}")
        
        diff = mix_img-train_covers
        diff = (diff-torch.min(diff))/(torch.max(diff)-torch.min(diff))
        toshow = torch.cat((train_secrets[:4]*_std_torch+_mean_torch,train_covers[:4],mix_img[:4],recover_vis),dim=0)
        imgg = make_grid(toshow,nrow=nrow_)
        save_image(imgg,'{}/testimages/{}.png'.format(outputname,idx),normalize=False)

    conn.send(b"DONE") 
    conn.close()
    server_socket.close()  
    print("Server closed - finished.")


if __name__ == "__main__":
    '''Constant declarations'''
    parser = argparse.ArgumentParser(description='Run training or testing on the network.')
    parser.add_argument('--train', action='store_true', help='Run in training mode.')
    parser.add_argument('--test', action='store_true', help='Run in testing mode.')
    parser.add_argument('--sal_for_he', action='store_true', help='Use saliency map to guide he or feature extraction.')
    parser.add_argument('--compress', action='store_true', help='Use compression.')
    parser.add_argument('--sal_for_cs', action='store_true', help='Use saliency map or feature extraction to guide compression.')
    parser.add_argument('--encrypt', action='store_true', help='Use encryption.')
    parser.add_argument('--base_project_path', type=str, help='Base path for the project.')
    parser.add_argument('--share_guide', action='store_true', help='If compression and encryption are used and they both use the same guidance method they may share the results at client side!')
    
    args = parser.parse_args()

    base_project_path = args.base_project_path
    train_mode = args.train
    test_mode = args.test
    small = args.sal_for_he
    compress = args.compress
    encrypt = args.encrypt
    sal_cs = args.sal_for_cs
    share_guide = args.share_guide

    #only use greyscale if only compress for comaprability
    if compress and not encrypt:
        color = False
    else:
        color = True

    #only use svhn if only encrypt
    if not compress and encrypt:
        svhn = True
    else:
        svhn = False

    # ------------------ Argument-Debug-Print ------------------
    print("-" * 50)
    print("Skript wird mit den folgenden Argumenten gestartet:")
    print(f"  Mode: {'Train' if train_mode else 'Test'}")
    print(f"  Project's path: {base_project_path}")
    print(f"  Compression: {'Yes' if compress else 'No'}")
    print(f"  Compression guided by: {'Saliency map' if sal_cs else 'Feature extraction'}")
    print(f"  Encryption: {'Yes' if encrypt else 'No'}")
    print(f"  Encryption guided by: {'Saliency map' if small else 'Feature extraction'}")
    print(f"  Shared Guidance: {'Yes' if share_guide else 'No'}")
    print("-" * 50)
    # -----------------------------------------------------------

    if small:
        batch_size = 64
    if not small:
        batch_size = 45
    batch_size_test = 10
    nrow_ = 4
    beta = 10
    kappa = 5
    learning_rate = 0.0001
    residual_en = True
    residual_de = True
    lambda_net = 0.8
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    output_base_dir = os.path.join(base_project_path, "output/", "experiments/")
    dataset_name_segment = "svhn" if svhn else "stl10"
    he_guide_segment = "saliency" if small else "feature extraction"
    cs_guide_segment = "saliency" if sal_cs else "feature extraction"
    compress_segment = "compressed" if compress else "uncompressed"
    encrypt_segment = "encrypted" if encrypt else "unencrypted"
    same_guide_segment = "shared" if share_guide else "seperate"
    color_segment = "color" if color else "grayscale"
    output_dir_name = (
        f"{dataset_name_segment}_"
        f"{encrypt_segment}_"
        f"{he_guide_segment}_"
        f"{compress_segment}_"
        f"{cs_guide_segment}_"
        f"{same_guide_segment}_"
        f"{color_segment}/"
    )
    outputname = os.path.join(output_base_dir, output_dir_name)
    os.makedirs(outputname, exist_ok=True)
    print(f"Output directory (Old files of the same flags will be overwritten!): {outputname}")

    MODELS_PATH = os.path.join(outputname, "model/")
    
    if svhn:
        target_model, ds_fetcher, is_imagenet = selector.select('svhn')
    
    else:
        DATA_ROOT = './data'
        os.makedirs(DATA_ROOT, exist_ok=True)
        target_model, ds_fetcher, is_imagenet = selector.select('stl10', model_root=DATA_ROOT)
    
    target_model.eval().to(device) 

    dummy_loader = ds_fetcher(batch_size=1, train=True, val=False)
    for images, _ in dummy_loader:
        imgsize = images.shape[2]
        break

    modellp = PerceptualSimilarity.models.PerceptualLoss(model='net-lin', net='alex', use_gpu=True, gpu_ids=[0])

    test_loader = ds_fetcher(batch_size=batch_size_test,train=False,val=True) 
    train_loader = ds_fetcher(batch_size=batch_size,train=True,val=False)
    _mean_torch = torch.tensor((0.5, 0.5, 0.5)).view(3,1,1).to(device)
    _std_torch = torch.tensor((0.5, 0.5, 0.5)).view(3,1,1).to(device)

    import glob
    checkpoint_pattern = os.path.join(outputname, '*.tar')
    checkpoint_files = glob.glob(checkpoint_pattern)

    if test_mode:
        test()
    if train_mode:
        train(train_loader, beta, learning_rate, batch_size, 0, 3)
    