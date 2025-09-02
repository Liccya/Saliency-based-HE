import torch.nn as nn
import torch
from torch.autograd import Variable
import juncw
import os
import torch.optim as optim

import torchvision.datasets as datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PrepNetwork(nn.Module):
    def __init__(self):
        super(PrepNetwork, self).__init__()
        self.initialP3 = nn.Sequential(
            nn.Conv2d(3, 50, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(50, 50, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(50, 50, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(50, 50, kernel_size=3, padding=1),
            nn.ReLU())
        self.initialP4 = nn.Sequential(
            nn.Conv2d(3, 50, kernel_size=4, padding=1),
            nn.ReLU(),
            nn.Conv2d(50, 50, kernel_size=4, padding=2),
            nn.ReLU(),
            nn.Conv2d(50, 50, kernel_size=4, padding=1),
            nn.ReLU(),
            nn.Conv2d(50, 50, kernel_size=4, padding=2),
            nn.ReLU())
        self.initialP5 = nn.Sequential(
            nn.Conv2d(3, 50, kernel_size=5, padding=2),
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
    def __init__(self,res=True,lambda_net=0.8):
        super(HidingNetwork, self).__init__()
        self.res = res
        self.lambda_net = lambda_net
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


    def forward(self, h,cover): # h is the 7-channel input, cover is the original 3-channel
        h1 = self.initialH3(h) # Takes 7-channel `h`
        h2 = self.initialH4(h) # Takes 7-channel `h`
        h3 = self.initialH5(h) # Takes 7-channel `h`
        mid = torch.cat((h1, h2, h3), 1)
        h4 = self.finalH3(mid)
        h5 = self.finalH4(mid)
        h6 = self.finalH5(mid)
        mid2 = torch.cat((h4, h5, h6), 1)
        out = self.finalH(mid2)

        if self.res:
            out = (1-self.lambda_net)*out + self.lambda_net*cover # Uses the 3-channel `cover` here
        return out


class RevealNetwork(nn.Module):
    def __init__(self,res=True,lambda_net =0.8):
        super(RevealNetwork, self).__init__()
        self.res = res
        self.lambda_net = lambda_net
        if self.res==True:
            in_c = 3
        else:
            in_c = 6
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

def getAdvZ(img_size, labels, minibs, target_model):
    c=10
    k=5
    st = 100
    z0 = torch.rand((minibs,3,img_size,img_size))
    cwatt = juncw.CW(target_model,c=c,kappa=k,steps=st,targeted=True, target_labels=labels)
    adv = cwatt(z0, labels)
    del cwatt

    return adv

# Join three networks in one module
class Netw(nn.Module):
    def __init__(self,resen=True,resde=True,lambda_net=0.8):
        super(Netw, self).__init__()
        self.resen = resen
        self.resde = resde
        self.lambda_net = lambda_net
        self.act = nn.Sigmoid()
        self.device = device # Ensure 'device' is defined globally or passed in
        
        # PrepNetwork (m1) is initialized but will be skipped in forward pass
        self.m1 = PrepNetwork().to(self.device) 
        
        # HidingNetwork (m2) expects a 7-channel input (secret_rgb + saliency + cover)
        self.m2 = HidingNetwork(resen, lambda_net).to(self.device)
        
        self.m3 = RevealNetwork(resde, lambda_net).to(self.device)

        # Removed optimizer init from here as it's done in train_client.py
        # learning_rate = 0.0001
        # optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.eval() # Set to evaluation mode by default, will be set to train in do_finetuning

        # It constructs the 7-channel input for m2 and passes the original cover for m2's second argument.
    def forward(self, secret_rgb, saliency_map, cover_img, train=True, **args): 
        #print(f"Shape of secret_rgb: {secret_rgb.shape}")
        #print(f"Shape of saliency_map: {saliency_map.shape}")
        hiding_input_tensor = torch.cat((secret_rgb, saliency_map), 1) # [B, 7, H, W]
        mix_img_before_quan = self.m2(hiding_input_tensor, cover_img)
        mix_img_after_quan = self.quan(mix_img_before_quan, type='noise', train=train)
        recovered_secret = self.m3(mix_img_after_quan, cover_img)

        return mix_img_after_quan, recovered_secret

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
    
from collections import OrderedDict   

def visualize_prep():
    # --- Visualization Code ---
    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(0) # Setze auch einen Seed für CPU-Operationen
    # Define transformations for SVHN
    transform = transforms.Compose([
        transforms.ToTensor(), # Converts PIL Image to Tensor (0-255 -> 0.0-1.0)
    ])

    # Load SVHN dataset
    svhn_dataset = datasets.SVHN(root='./data', split='test', download=True, transform=transform)

    # Get one sample image
    sample_idx = 0
    sample_image_tensor, label = svhn_dataset[sample_idx]

    # Add a batch dimension (PrepNetwork expects batch input)
    input_image_batch = sample_image_tensor.unsqueeze(0).to(device)

    # Instantiate PrepNetwork and move to device
    prep_net = PrepNetwork().to(device)
    
    
    checkpoint_path = '/home/yvonne/Documents/CNN Project/RIC/Weights/Svhn.pth.tar' # Oder MODELS_PATH_small

    if os.path.isfile(checkpoint_path):
        print(f"Lade Gewichte von {checkpoint_path}...")
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
        
        # Remove 'module.' prefix if it exists (from DataParallel, common in saved checkpoints)
        filtered_prep_net_state_dict = OrderedDict()
        for k, v in prep_net_state_dict.items():
            name = k[7:] if k.startswith('module.') else k
            filtered_prep_net_state_dict[name] = v

        try:
            prep_net.load_state_dict(filtered_prep_net_state_dict)
            print("PrepNetwork Gewichte erfolgreich geladen.")
        except RuntimeError as e:
            print(f"Fehler beim Laden der PrepNetwork Gewichte: {e}")
            print("Stelle sicher, dass der Checkpoint die korrekten PrepNetwork Gewichte enthält und die Architektur übereinstimmt.")
    else:
        print(f"Warnung: Kein Checkpoint unter '{checkpoint_path}' gefunden. PrepNetwork wird mit zufälligen Gewichten ausgeführt.")

    prep_net.eval() # Set to evaluation mode

    # Perform forward pass
    with torch.no_grad():
        feature_maps = prep_net(input_image_batch)

    # Extract feature maps for visualization
    feature_maps = feature_maps.squeeze(0)

    # Convert original image to numpy for display
    # HIER IST DIE KORREKTUR: .cpu() HINZUFÜGEN
    original_image_np = sample_image_tensor.permute(1, 2, 0).cpu().numpy()

    # Prepare feature maps for visualization
    num_feature_maps_to_display = 10 # Display first 5 feature maps
    fig, axes = plt.subplots(1, num_feature_maps_to_display + 2, figsize=(300, 3)) # +2 for original and mean

    # Display original image
    axes[0].imshow(original_image_np)
    axes[0].set_title(f'Original SVHN Image\n(Label: {label})')
    axes[0].axis('off')

    # Display individual feature maps
    for i in range(num_feature_maps_to_display):
        # Get a single feature map (32x32)
        # HIER IST DIE KORREKTUR: .cpu() HINZUFÜGEN
        feature_map_np = feature_maps[i].cpu().numpy()
        
        # Normalize feature map for display (0-1 range)
        min_val = feature_map_np.min()
        max_val = feature_map_np.max()
        if max_val - min_val > 0:
            normalized_feature_map = (feature_map_np - min_val) / (max_val - min_val)
        else:
            normalized_feature_map = np.zeros_like(feature_map_np)
        
        axes[i+1].imshow(normalized_feature_map, cmap='viridis')
        axes[i+1].set_title(f'Feature Map {i+1}')
        axes[i+1].axis('off')

    # Display the mean feature map
    # HIER IST DIE KORREKTUR: .cpu() HINZUFÜGEN
    mean_feature_map_np = torch.mean(feature_maps, dim=0).cpu().numpy()
    min_val_mean = mean_feature_map_np.min()
    max_val_mean = mean_feature_map_np.max()
    if max_val_mean - min_val_mean > 0:
        normalized_mean_feature_map = (mean_feature_map_np - min_val_mean) / (max_val_mean - min_val_mean)
    else:
        normalized_mean_feature_map = np.zeros_like(mean_feature_map_np)

    axes[num_feature_maps_to_display + 1].imshow(normalized_mean_feature_map, cmap='viridis')
    axes[num_feature_maps_to_display + 1].set_title('Mean Feature Map')
    axes[num_feature_maps_to_display + 1].axis('off')

    plt.tight_layout()
    plt.suptitle('PrepNetwork Output Feature Maps (First 5 and Mean)', y=1.05, fontsize=16)
    plt.show()

def visualize_prep_stl10():
    # --- Visualization Code ---
    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(0) # Setze auch einen Seed für CPU-Operationen

    # Define transformations for STL10
    transform = transforms.Compose([
        transforms.ToTensor(), # Converts PIL Image to Tensor (0-255 -> 0.0-1.0)
        # HINZUFÜGEN: Skaliere auf die Größe, die deine PrepNetwork erwartet
        # Wenn dein Modell für 32x32 Bilder trainiert wurde, uncomment diese Zeile:
        transforms.Resize((32, 32)),
    ])

    # Load STL10 dataset
    # ÄNDERUNG HIER: datasets.SVHN -> datasets.STL10
    stl10_dataset = datasets.STL10(root='./data', split='test', download=True, transform=transform)

    # Get one sample image
    sample_idx = 0
    sample_image_tensor, label = stl10_dataset[sample_idx]

    # Add a batch dimension (PrepNetwork expects batch input)
    input_image_batch = sample_image_tensor.unsqueeze(0).to(device)

    # Instantiate PrepNetwork and move to device
    prep_net = PrepNetwork().to(device)
    
    
    checkpoint_path = '/home/yvonne/Documents/CNN Project/RIC/Weights/Svhn.pth.tar' # Oder MODELS_PATH_small

    if os.path.isfile(checkpoint_path):
        print(f"Lade Gewichte von {checkpoint_path}...")
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
        
        # Remove 'module.' prefix if it exists (from DataParallel, common in saved checkpoints)
        filtered_prep_net_state_dict = OrderedDict()
        for k, v in prep_net_state_dict.items():
            name = k[7:] if k.startswith('module.') else k
            filtered_prep_net_state_dict[name] = v

        try:
            prep_net.load_state_dict(filtered_prep_net_state_dict)
            print("PrepNetwork Gewichte erfolgreich geladen.")
        except RuntimeError as e:
            print(f"Fehler beim Laden der PrepNetwork Gewichte: {e}")
            print("Stelle sicher, dass der Checkpoint die korrekten PrepNetwork Gewichte enthält und die Architektur übereinstimmt.")
    else:
        print(f"Warnung: Kein Checkpoint unter '{checkpoint_path}' gefunden. PrepNetwork wird mit zufälligen Gewichten ausgeführt.")

    prep_net.eval() # Set to evaluation mode

    # Perform forward pass
    with torch.no_grad():
        feature_maps = prep_net(input_image_batch)

    # Extract feature maps for visualization
    feature_maps = feature_maps.squeeze(0)

    # Convert original image to numpy for display
    # HIER IST DIE KORREKTUR: .cpu() HINZUFÜGEN
    original_image_np = sample_image_tensor.permute(1, 2, 0).cpu().numpy()

    # Prepare feature maps for visualization
    num_feature_maps_to_display = 10 # Display first 5 feature maps
    fig, axes = plt.subplots(1, num_feature_maps_to_display + 2, figsize=(300, 3)) # +2 for original and mean

    # Display original image
    axes[0].imshow(original_image_np)
    axes[0].set_title(f'Original SVHN Image\n(Label: {label})')
    axes[0].axis('off')

    # Display individual feature maps
    for i in range(num_feature_maps_to_display):
        # Get a single feature map (32x32)
        # HIER IST DIE KORREKTUR: .cpu() HINZUFÜGEN
        feature_map_np = feature_maps[i].cpu().numpy()
        
        # Normalize feature map for display (0-1 range)
        min_val = feature_map_np.min()
        max_val = feature_map_np.max()
        if max_val - min_val > 0:
            normalized_feature_map = (feature_map_np - min_val) / (max_val - min_val)
        else:
            normalized_feature_map = np.zeros_like(feature_map_np)
        
        axes[i+1].imshow(normalized_feature_map, cmap='viridis')
        axes[i+1].set_title(f'Feature Map {i+1}')
        axes[i+1].axis('off')

    # Display the mean feature map
    # HIER IST DIE KORREKTUR: .cpu() HINZUFÜGEN
    mean_feature_map_np = torch.mean(feature_maps, dim=0).cpu().numpy()
    min_val_mean = mean_feature_map_np.min()
    max_val_mean = mean_feature_map_np.max()
    if max_val_mean - min_val_mean > 0:
        normalized_mean_feature_map = (mean_feature_map_np - min_val_mean) / (max_val_mean - min_val_mean)
    else:
        normalized_mean_feature_map = np.zeros_like(mean_feature_map_np)

    axes[num_feature_maps_to_display + 1].imshow(normalized_mean_feature_map, cmap='viridis')
    axes[num_feature_maps_to_display + 1].set_title('Mean Feature Map')
    axes[num_feature_maps_to_display + 1].axis('off')

    plt.tight_layout()
    plt.suptitle('PrepNetwork Output Feature Maps (First 5 and Mean)', y=1.05, fontsize=16)
    plt.show()

def visualize_hiding():
    # ---- Load Networks ----
    prep_net = PrepNetwork().to(device).eval()
    hiding_net = HidingNetwork().to(device).eval()

    # ---- Load SVHN sample ----
    transform = transforms.ToTensor()
    svhn_dataset = datasets.SVHN(root='./data', split='test', download=True, transform=transform)

    # Use two images: one as secret, one as cover
    secret_img_tensor, _ = svhn_dataset[0]
    cover_img_tensor, _ = svhn_dataset[1]

    # Add batch dimension
    secret_img_batch = secret_img_tensor.unsqueeze(0).to(device)
    cover_img_batch = cover_img_tensor.unsqueeze(0).to(device)

    # ---- Forward Pass through the Network ----
    with torch.no_grad():
        encoded_features = prep_net(secret_img_batch)
        
        # Original hiding network
        concat_input = torch.cat((encoded_features, cover_img_batch), dim=1)
        hidden_img_original = hiding_net(concat_input, cover_img_batch)

    # ---- Convert tensors to numpy for display ----
    def to_numpy_img(tensor_img):
        img = tensor_img.squeeze(0).detach().to(device).numpy()
        img = np.clip(img.transpose(1, 2, 0), 0, 1)  # (C, H, W) -> (H, W, C)
        return img

    secret_np = to_numpy_img(secret_img_batch)
    cover_np = to_numpy_img(cover_img_batch)
    hidden_np_original = to_numpy_img(hidden_img_original)

    # ---- Plotting ----
    fig, axes = plt.subplots(1, 3, figsize=(16, 3))
    axes[0].imshow(secret_np)
    axes[0].set_title('Secret Image')
    axes[0].axis('off')

    axes[1].imshow(cover_np)
    axes[1].set_title('Cover Image')
    axes[1].axis('off')

    axes[2].imshow(hidden_np_original)
    axes[2].set_title('HidingNetwork Output')
    axes[2].axis('off')


    plt.tight_layout()
    plt.suptitle('Comparison: Original vs Small Hiding Network', y=1.05, fontsize=16)
    plt.show()

#visualize_prep_stl10()