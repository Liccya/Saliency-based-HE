import torch.nn as nn
import torch
import random
import io, os, time
from decompress_model import Adaptive_CS

class RevealNetwork(nn.Module):
    def __init__(self, res=True,lambda_net =0.8):
        super(RevealNetwork, self).__init__()
        self.res = res
        self.lambda_net = lambda_net
        self.initialR3 = nn.Sequential(
            nn.Conv2d(3, 50, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(50, 50, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(50, 50, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(50, 50, kernel_size=3, padding=1),
            nn.ReLU())
        self.initialR4 = nn.Sequential(
            nn.Conv2d(3, 50, kernel_size=4, padding=1),
            nn.ReLU(),
            nn.Conv2d(50, 50, kernel_size=4, padding=2),
            nn.ReLU(),
            nn.Conv2d(50, 50, kernel_size=4, padding=1),
            nn.ReLU(),
            nn.Conv2d(50, 50, kernel_size=4, padding=2),
            nn.ReLU())
        self.initialR5 = nn.Sequential(
            nn.Conv2d(3, 50, kernel_size=5, padding=2),
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

def send_tensor(sock, tensor):
    buffer = io.BytesIO()
    torch.save(tensor, buffer)
    sock.sendall(len(buffer.getvalue()).to_bytes(4, 'big'))  # send length first
    sock.sendall(buffer.getvalue())

def receive_tensor(sock):
    map_location='cuda'
    length_bytes = sock.recv(4)
    length = int.from_bytes(length_bytes, 'big')
    buffer = b""
    while len(buffer) < length:
        buffer += sock.recv(length - len(buffer))
    tensor = torch.load(io.BytesIO(buffer), map_location=map_location)
    return tensor


# Join three networks in one module
class Net(nn.Module):
    def __init__(self,base_project_path, resen=True,resde=True,lambda_net=0.8, compress =True, encrypt = True):
        super(Net, self).__init__()
        self.resen = resen
        self.resde = resde
        self.lambda_net = lambda_net
        self.encrypt = encrypt
        self.compress = compress
        self.m3 = RevealNetwork(resde,lambda_net)
        self.act = nn.Sigmoid()
        #only use greyscale if only compress for comaprability
        if compress and not encrypt:
            color = False
        else:
            color = True


        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if self.compress:
            epoch = 320
            self.N_p = 13
            B = 8
            N = B*B

            model = Adaptive_CS(base_project_path, color, self.N_p, B, 1, torch.zeros(N, N))
            model = torch.nn.DataParallel(model).to(device)
            model_dir = '%s/layer_%d_block_%d' % (os.path.join(base_project_path, 'compress_Data/model'), self.N_p, B)
            #print("DEBUG: trying to load model from: ", model_dir)
            model.load_state_dict(torch.load('%s/net_params_%d.pkl' % (model_dir, epoch), map_location=device))
            self.model = model


    def forward(self, secret, cover, labels, conn, train = False):
        if not train:
            start_time = time.time()
            send_tensor(conn, secret)
            send_tensor(conn, cover)
            send_tensor(conn, labels)
            
            input_ = receive_tensor(conn)
            mem = receive_tensor(conn)
            end_time = time.time()
            duration = end_time-start_time

            saved_data = 0
            #flawed because too much needed to be changed in communication for the scripts to work... no meaningful data saving possible... but other papers already show feasability of this so its ok
            '''if self.compress:
                #calculate compression rate
                saved_data_list = []
                for idx, i in enumerate(input_):
                    input_size_mb = secret[idx].element_size() * secret[idx].numel() / (1024**2)
                    PhiT_Phi_basic, PhiT_y, x_uv_downsampled, L, shape_info = i
                    phi_t_y_basic_size_mb = PhiT_y.element_size() * PhiT_y.numel() / (1024**2)
                    phi_t_phi_basic_size_mb = PhiT_Phi_basic.element_size() * PhiT_Phi_basic.numel() / (1024**2)
                    if x_uv_downsampled is not 0:
                        x_uv_size_mb = x_uv_downsampled.element_size() * x_uv_downsampled.numel() / (1024**2)
                    else:
                        x_uv_size_mb = 0
                    total_compressed_size_mb = phi_t_y_basic_size_mb + phi_t_phi_basic_size_mb + x_uv_size_mb
                    saved_data_list.append(input_size_mb - total_compressed_size_mb)
                    print(f'DEBUG: input_size_mb {secret[idx].element_size()}, phi_t_y_basic_size_mb {PhiT_y.element_size()}, phi_t_phi_basic_size_mb {phi_t_phi_basic_size_mb}, x_uv_size_mb {x_uv_size_mb}')
                saved_data=sum(saved_data_list)/len(saved_data_list)'''

        if train:
            input_ = secret
            duration = 0
            saved_data = 0
            mem = None
        #duration = receive_tensor(conn)
        if self.compress and not train:
            reconstructed_images = []
            rand_modes = [random.randint(0, 7) for _ in range(self.N_p)]
            for i in input_:
                x_2_single = self.model(
                        rand_modes,
                        i
                    )
                reconstructed_images.append(x_2_single)
            x_2= torch.cat(reconstructed_images, dim=0)
        else:
            x_2 = input_

        if self.encrypt:
            x_3 = self.m3(x_2,cover)
        else:
            x_3 = x_2

        return x_2, x_3, duration, mem, saved_data
