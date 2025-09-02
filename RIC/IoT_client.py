import socket
import torch
import io
import os
from IoT_small import Netw as Netw_small
from IoT_regular import Netw as Netw_regular
from IoT_compress import Adaptive_CS 
import torchvision.transforms as transforms
from collections import OrderedDict
from pre_process_image import process_image, init_model, generate_grad_cam_overlay 
from PIL import Image 
import numpy as np 
import argparse
import tracemalloc


def send_tensor(sock, tensor):
    buffer = io.BytesIO()
    torch.save(tensor, buffer)
    sock.sendall(len(buffer.getvalue()).to_bytes(4, 'big'))
    sock.sendall(buffer.getvalue())

def receive_tensor(sock):
    map_location = 'cpu'
    length_bytes = sock.recv(4)
    if not length_bytes: return None
    length = int.from_bytes(length_bytes, 'big')
    buffer = b''
    while len(buffer) < length:
        data = sock.recv(length - len(buffer))
        if not data: return None
        buffer += data
    tensor = torch.load(io.BytesIO(buffer), map_location=map_location)
    return tensor


def load_resume_checkpoint(path, model, target_device, m1):

    filepath = os.path.join(path, 'hide_network_weights.pth')
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
            model.m2.load_state_dict(filtered_model_state_dict, strict=True)
        except RuntimeError as e:
            print(f"  WARNING: Error loading model state_dict strictly: {e}")
            print("  Attempting to load with strict=False (some keys might be missing/mismatched).")
            model.load_state_dict(filtered_model_state_dict, strict=False)

        epoch = checkpoint.get('epoch', 0)
        model.to(target_device)
        print(f"=> Resumed from epoch {epoch} and model moved to {target_device}.")

        if m1:
            filepath = os.path.join(path, 'prep_network_weights.pth')
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
                    model.m1.load_state_dict(filtered_model_state_dict, strict=True)
                except RuntimeError as e:
                    print(f"  WARNING: Error loading model state_dict strictly: {e}")
                    print("  Attempting to load with strict=False (some keys might be missing/mismatched).")
                    model.load_state_dict(filtered_model_state_dict, strict=False)

                epoch = checkpoint.get('epoch', 0)
                model.to(target_device)
                print(f"=> Resumed from epoch {epoch} and model moved to {target_device}.")
            else:
                print(f"=> No checkpoint found at '{filepath}'. Nothing loaded.")
                return model,  0

        return model, epoch

    else:
        print(f"=> No checkpoint found at '{filepath}'. Nothing loaded.")
        return model,  0

def get_saliency(secret):
    
    mean_tensor_denorm = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1).cpu()
    std_tensor_denorm = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1).cpu()

    pre_model = init_model().to(client_net_device)
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
        
        single_saliency_tensor = norm_transform_for_cropped(heatmap_pil_img).to(client_net_device)
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

def script():
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        print("Connecting to Server...")
        client_socket.connect((HOST, PORT))
        print(f"Connected to {HOST}:{PORT}")
    except ConnectionRefusedError:
        print(f"Connection could not be established")
        exit()
    
    small, compress, encrypt, sal_for_cs, share_guide, outname, base_project_path = receive_tensor(client_socket)
    if client_base_project_path == '.':
        outputname = outname
    else:
        # Finde den Index des Beginns von '/output/' im Pfad
        index_von_output = outname.find('/output/')

        # Wenn '/output/' gefunden wurde (index ist nicht -1)
        if index_von_output != -1:
            # Nimm den Teil des Pfades, der bei '/output/' anfängt
            restlicher_pfad = outname[index_von_output:]
            
            # Füge den neuen Basispfad davor
            outputname = f"{client_base_project_path}{restlicher_pfad}"
        else:
            # Falls kein '/output/' gefunden wird, verwenden wir den kompletten outname
            outputname = f"{client_base_project_path}{outname}"
        base_project_path = client_base_project_path

    #only use greyscale if only compress for comaprability
    if compress and not encrypt:
        color = False
    else:
        color = True
    

    MODELS_PATH = os.path.join(outputname, "model/")

    if encrypt:
        if small:
            net = Netw_small(residual_en,residual_de,lambda_net).to(client_net_device)
            net, _ = load_resume_checkpoint(MODELS_PATH, net, client_net_device, False)
        else:
            net = Netw_regular(residual_en,residual_de,lambda_net).to(client_net_device)
            net, _ = load_resume_checkpoint(MODELS_PATH, net, client_net_device, True)
    
    if compress:
        model = Adaptive_CS(base_project_path, sal_for_cs, color, N_p, B, 1, torch.zeros(N, N))
        model = torch.nn.DataParallel(model).to(client_net_device)
        model_dir = '%s/layer_%d_block_%d' % (os.path.join(base_project_path,'compress_Data/model'), N_p, B)
        model.load_state_dict(torch.load('%s/net_params_%d.pkl' % (model_dir, 320), map_location=client_net_device))


    while True:
        try:
            tracemalloc.start()
            secret = receive_tensor(client_socket)
            if secret is None:
                print("Server connection lost or no data left to compute.")
                break
            cover = receive_tensor(client_socket)
            labels = receive_tensor(client_socket)
            labels = labels.to(client_net_device)

            secret_input_for_net = secret.to(client_net_device)
            cover_tensor = cover.to(client_net_device)

            if encrypt:
                if small:
                    secret_sal=get_saliency(secret)
                else:
                    secret_sal = None
                with torch.no_grad():
                    #print(f"DEBUG: Client input shape: {secret.shape}")
                    x_2, share = net(secret_input_for_net, secret_sal, cover_tensor, train=False)
                #print(f"DEBUG: Client x_2 shape: {x_2.shape}")
            else: 
                x_2 = secret_input_for_net

            if share_guide:
                if small is not sal_for_cs:
                    raise ValueError("If guidance shall be shared the compression and encryption need to have the same guidance!")
            else:
                share = None
            
            if compress:
                x_2_compressed=[]
                for i in range(x_2.shape[0]):
                    if share is not None:
                        # Extract the single saliency map for the current image
                        share_single = share[i:i+1]
                    else:
                        share_single = None
                    x_2_single = x_2[i:i+1]
                    PhiT_Phi_basic, PhiT_y_basic, x_uv_downsampled, L, shape_info = model(
                    x_2_single, 
                    int(np.ceil(0.01 * N)),
                    0.8,
                    0.3,
                    share_single
                    )   
                    x_2_single_compressed=[PhiT_Phi_basic, PhiT_y_basic, x_uv_downsampled, L,shape_info]
                    x_2_compressed.append(x_2_single_compressed)
                x_2 = x_2_compressed
            #print(f"DEBUG: Client x_2 0 after compress: {x_2[0]}")
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            send_tensor(client_socket, x_2)
            send_tensor(client_socket, [current,peak])
            

        except (ConnectionResetError, BrokenPipeError):
            print("Connection to Server lost.")
            break
        except Exception as e:
            print(f"An Error occured: {e}")
            break

    client_socket.close()
    print("Client shut down.")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run training or testing on the network.')
    parser.add_argument('--base_project_path', type=str, default='.', help='Base path for the project.')
    args = parser.parse_args()
    client_base_project_path = args.base_project_path

    residual_en = True
    residual_de = True
    lambda_net = 0.8
    HOST = '127.0.0.1' 
    PORT = 5001
    client_net_device = torch.device('cuda' if torch.cuda.is_available() else  "cpu")
    N_p = 13
    B = 8
    N = B*B

    script()
