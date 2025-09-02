import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import os
from PIL import Image
import cv2
import numpy as np
import pandas as pd
from io import BytesIO
from torchvision.utils import make_grid,save_image
from torch.utils.data import Dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Transforms (bleiben gleich)
norm_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                        [0.229, 0.224, 0.225])
])

def process_image(pil_img_input, model):
    overlay_img, raw_heatmap_for_cropping = generate_grad_cam_overlay(model, pil_img_input)
    cropped_original_img = resize_image(pil_img_input, raw_heatmap_for_cropping, th=0.5)
    #print_image(pil_img_input) 
    
    return cropped_original_img


def print_image(img):
    outputname="image_processed"
    if not os.path.exists(outputname):
        os.mkdir(outputname)
    idx = 0
    while os.path.exists('{}/{}.png'.format(outputname,idx)):
        idx = idx+1

    mean_tensor = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1).cpu()
    std_tensor = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1).cpu()
    
    if isinstance(img, torch.Tensor):
        img_denormalized = img.clone().detach().cpu() * std_tensor + mean_tensor
        img_denormalized = torch.clamp(img_denormalized, 0, 1)
        toshow = img_denormalized[:4] # Annahme, dass es ein Batch von Bildern ist
        imgg = make_grid(toshow, nrow=1)
        save_image(imgg,'{}/{}.png'.format(outputname,idx),normalize=False)
    elif isinstance(img, Image.Image):
        img.save('{}/{}.png'.format(outputname,idx))
    else:
        print("Unsupported image type for print_image")


def train(project_path, dir_info_name):
    data_root_dir = os.path.join(project_path, "data", "svhn")

    try:
        train_dataset = datasets.SVHN(root=data_root_dir, split='train', download=True, transform=norm_transform)
    except Exception as e:
        print(f"Fehler beim Laden des SVHN-Trainingsdatensatzes: {e}")
        print(f"Stelle sicher, dass der Pfad '{data_root_dir}' existiert und beschreibbar ist.")
        return

    if len(train_dataset) == 0:
        print(f"Keine Daten im Trainingsdatensatz gefunden unter: {data_root_dir}")
        return

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=torch.cuda.is_available())

    model = models.mobilenet_v3_small(weights=None) # pretrained=True ist nicht nötig, da wir es später laden
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, 10)
    model = model.to('cuda' if torch.cuda.is_available() else 'cpu')

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    loss_history = []
    acc_history = []
    for epoch in range(10):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to('cuda' if torch.cuda.is_available() else 'cpu'), labels.to('cuda' if torch.cuda.is_available() else 'cpu')

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, preds = outputs.max(1)
            correct += preds.eq(labels).sum().item()
            total += labels.size(0)

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = correct / total if total > 0 else 0.0

        loss_history.append(epoch_loss)
        acc_history.append(epoch_acc)

        print(f"Epoch {epoch+1}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}")

    save_path = os.path.join(project_path, f"mobilenet_finetuned_{dir_info_name}.pth")
    torch.save(model.state_dict(), save_path)
    print(f"Weights saved under {save_path}")


def generate_grad_cam_overlay(model, original_pil_img):
    img_tensor = norm_transform(original_pil_img).unsqueeze(0).to(device)
    target_layer=model.features[-1]
    model.eval()
    img_tensor.requires_grad_(True) 

    activations = []
    gradients = []

    def forward_hook(module, input, output):
        output.requires_grad_(True) 
        activations.append(output)

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    forward_handle = target_layer.register_forward_hook(forward_hook)
    backward_handle = target_layer.register_backward_hook(backward_hook)
    
    output = model(img_tensor)
    pred_class = output.argmax(dim=1).item()
    
    model.zero_grad()
    output[0, pred_class].backward()

    forward_handle.remove()
    backward_handle.remove()

    if not activations or not gradients:
        placeholder_overlay = Image.fromarray(np.zeros(original_pil_img.size + (3,), dtype=np.uint8))
        placeholder_heatmap = np.zeros(original_pil_img.size[::-1], dtype=np.float32)
        return placeholder_overlay, placeholder_heatmap

    weights = torch.mean(gradients[0], dim=[2, 3], keepdim=True)
    heatmap = torch.sum(weights * activations[0], dim=1).squeeze()
    heatmap = np.maximum(heatmap.cpu().detach().numpy(), 0)
    if np.max(heatmap) > 0:
        heatmap /= np.max(heatmap)
    else:
        heatmap = np.zeros_like(heatmap)

    raw_heatmap_resized = cv2.resize(heatmap, (original_pil_img.size[0], original_pil_img.size[1]))
    
    img_np = np.array(original_pil_img.convert('RGB'))
    heatmap_for_overlay = cv2.resize(heatmap, (img_np.shape[1], img_np.shape[0]))
    heatmap_for_overlay = cv2.applyColorMap(np.uint8(255 * heatmap_for_overlay), cv2.COLORMAP_JET)
    heatmap_for_overlay = cv2.cvtColor(heatmap_for_overlay, cv2.COLOR_BGR2RGB)
    superimposed_img_np = cv2.addWeighted(img_np, 0.6, heatmap_for_overlay, 0.4, 0)

    return Image.fromarray(superimposed_img_np), raw_heatmap_resized

def resize_image(img_pil, sal_heatmap_np, th=0.5):
    img_np = np.array(img_pil)
    if sal_heatmap_np.ndim == 3:
        sal_heatmap_np = np.mean(sal_heatmap_np, axis=2)
    h, w = sal_heatmap_np.shape[:2]
    y_min, y_max, x_min, x_max = h, 0, w, 0
    rows_with_salience = np.any(sal_heatmap_np > th, axis=1)
    if not np.any(rows_with_salience):
        return Image.fromarray(np.zeros_like(img_np))
    y_min = np.where(rows_with_salience)[0][0]
    y_max = np.where(rows_with_salience)[0][-1] + 1
    cols_with_salience = np.any(sal_heatmap_np > th, axis=0)
    x_min = np.where(cols_with_salience)[0][0]
    x_max = np.where(cols_with_salience)[0][-1] + 1
    if x_min >= x_max or y_min >= y_max:
        return Image.fromarray(np.zeros_like(img_np))
    cropped_img_np = img_np[y_min:y_max, x_min:x_max]
    return Image.fromarray(cropped_img_np)

def init_model():
    save_path = os.path.join(current_project_path, "mobilenet_finetuned_svhn_custom_model.pth")

    model = models.mobilenet_v3_small(weights=None)
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, 10)
    model.load_state_dict(torch.load(save_path, map_location=device))
    model.eval().to(device)
    return model

def test():
    data_root_dir = os.path.join(current_project_path, "data", "svhn")
    model = init_model()
    
    # Identifiziere die letzte Faltungsschicht
    # Hier können Sie den target_layer anpassen, z.B. zu model.features[11] oder model.features[9]
    # Empfehlung: Probieren Sie model.features[11] oder model.features[9], wenn die Saliency Maps einfarbig sind
    target_layer = model.features[-1] # Dies ist model.features[12] bei mobilenet_v3_small
    # target_layer = model.features[11] # Beispiel: einen früheren Layer wählen
    # target_layer = model.features[9]  # Beispiel: noch einen früheren Layer wählen


    try:
        test_dataset = datasets.SVHN(root=data_root_dir, split='test', download=True, transform=norm_transform)
    except Exception as e:
        print(f"Fehler beim Laden des SVHN-Testdatensatzes: {e}")
        print(f"Stelle sicher, dass der Pfad '{data_root_dir}' existiert und beschreibbar ist.")
        return

    if len(test_dataset) == 0:
        print(f"Keine Daten im Testdatensatz gefunden unter: {data_root_dir}")
        return

    # test_loader bleibt batch_size=1 für Grad-CAM
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # Erstelle die Ausgabeordner
    output_grad_cam_dir = os.path.join(current_project_path, "grad_cam_outputs")
    output_cropped_dir = os.path.join(current_project_path, "cropped_images")
    output_raw_saliency_dir = os.path.join(current_project_path, "debug_raw_saliency")

    os.makedirs(output_grad_cam_dir, exist_ok=True)
    os.makedirs(output_cropped_dir, exist_ok=True)
    os.makedirs(output_raw_saliency_dir, exist_ok=True)


    print(f"Generiere Grad-CAM Overlays und speichere sie in: {output_grad_cam_dir}")
    print(f"Generiere zugeschnittene Bilder und speichere sie in: {output_cropped_dir}")
    print(f"Speichere rohe Saliency Maps in: {output_raw_saliency_dir}")

    for i, (inputs, labels) in enumerate(test_loader):
        if i >=100:
            print("Maximum von 100 Bildern für den Test erreicht. Beende Testschleife.")
            break

        inputs_on_device = inputs.to(device) # Tensor für das Modell

        # Denormalisiertes PIL-Bild für das Overlay
        mean_tensor = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1).cpu()
        std_tensor = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1).cpu()
        img_denormalized_tensor = inputs[0].cpu() * std_tensor + mean_tensor
        original_pil_img = transforms.ToPILImage()(img_denormalized_tensor.clamp(0,1))

        # Vorhersage des Modells
        model.eval()
        with torch.no_grad(): # No-grad hier, da wir nur die Vorhersage brauchen, nicht Gradienten dafür
            outputs = model(inputs_on_device)
            _, predicted = torch.max(outputs.data, 1)

        # Generiere Grad-CAM Overlay und die rohe Heatmap
        overlay_img, raw_heatmap_for_cropping = generate_grad_cam_overlay(model, inputs_on_device, original_pil_img, target_layer)

        # Dateinamen erstellen: index_trueLabel_predictedLabel
        base_filename = f"{i+1}_true_{labels.item()}_pred_{predicted.item()}"

        # Speichern des Grad-CAM Overlays
        save_path_cam = os.path.join(output_grad_cam_dir, f"{base_filename}_cam.png")
        overlay_img.save(save_path_cam)
        print(f"Saved CAM: {save_path_cam}")

        # Speichern der rohen Saliency Map
        # Muss in ein PIL Image konvertiert werden, bevor es gespeichert wird
        # Die raw_heatmap_for_cropping ist bereits auf Originalgröße skaliert und 0-1 normiert.
        raw_saliency_pil = Image.fromarray(np.uint8(255 * raw_heatmap_for_cropping)) # Skaliere 0-1 auf 0-255 für Bildspeicherung
        save_path_raw_saliency = os.path.join(output_raw_saliency_dir, f"{base_filename}_debug_raw_saliency.png")
        raw_saliency_pil.save(save_path_raw_saliency)
        print(f"Saved Debug Raw Saliency: {save_path_raw_saliency}")

        # Speichern des zugeschnittenen Originalbildes basierend auf der Saliency Map
        cropped_original_img = resize_image(original_pil_img, raw_heatmap_for_cropping, th=0.5)
        save_path_cropped = os.path.join(output_cropped_dir, f"{base_filename}_cropped.png")
        cropped_original_img.save(save_path_cropped)
        print(f"Saved Cropped: {save_path_cropped}")

        # Speichern des Originalbildes zum Vergleich (optional, aber nützlich)
        save_path_original_comp = os.path.join(output_cropped_dir, f"{base_filename}_original.png")
        original_pil_img.save(save_path_original_comp)
        print(f"Saved Original (for comparison): {save_path_original_comp}")

    print("Alle Grad-CAM Overlays und zugeschnittenen Bilder generiert und gespeichert.")


current_project_path = "/home/yvonne/Documents/CNN Project/"
directory_info = {'name': 'svhn_custom_model'}

# Run test and create overlay
#test()
#train(current_project_path, directory_info['name'])