import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataset import LoRaDataset
from models.probabilistic import ProbabilisticMultiTaskResNet
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np
import os
import sys
import warnings


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import KNN_ANOMALY_THRESHOLD

warnings.filterwarnings('ignore')

def extract_features(model, x):
    x = x.view(-1, 1, 128, 128)
    x = F.relu(model.bn1(model.conv1(x)))
    x = model.layer1(x)
    x = model.layer2(x)
    x = model.layer3(x)
    x = model.attention(x)
    x = model.pool(x)
    return x.view(x.size(0), -1)

def generate_plots():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- GENERATING VISUALIZATIONS ON {str(device).upper()} ---")

    
    model = ProbabilisticMultiTaskResNet(num_devices=30).to(device)
    model.load_state_dict(torch.load("lora_model_hardened.pth", map_location=device, weights_only=True))
    model.eval()

    print("Extracting Legitimate Features...")
    legit_dataset = LoRaDataset(r'data\raw\Train\dataset_training_no_aug.h5', subset_size=100)
    legit_loader = DataLoader(legit_dataset, batch_size=100, shuffle=True)
    legit_x, legit_y, _ = next(iter(legit_loader))
    
    with torch.no_grad():
        legit_features = extract_features(model, legit_x.to(device))

    print("Extracting Rogue Features...")
   
    rogue_dataset = LoRaDataset(r'data\raw\Test\dataset_rogue.h5', subset_size=50, is_rogue_dataset=True)
    rogue_loader = DataLoader(rogue_dataset, batch_size=50, shuffle=True)
    rogue_x, _, _ = next(iter(rogue_loader))
    
    with torch.no_grad():
        rogue_features = extract_features(model, rogue_x.to(device))

    print("Generating Distance Histogram...")
    distances_matrix = torch.cdist(rogue_features, legit_features)
    rogue_min_distances, _ = torch.min(distances_matrix, dim=1)
    
    legit_internal = torch.cdist(legit_features, legit_features)
    legit_internal.fill_diagonal_(float('inf')) 
    legit_min_distances, _ = torch.min(legit_internal, dim=1)

    plt.figure(figsize=(10, 6))
    plt.hist(legit_min_distances.cpu().numpy(), bins=20, alpha=0.7, color='blue', label='Legitimate Signals')
    plt.hist(rogue_min_distances.cpu().numpy(), bins=20, alpha=0.7, color='red', label='Zero-Day Rogue Signals')
    
    
    plt.axvline(x=KNN_ANOMALY_THRESHOLD, color='black', linestyle='dashed', linewidth=2, label=f'Security Threshold ({KNN_ANOMALY_THRESHOLD})')
    
    plt.title('Feature-Space Anomaly Detection (1-NN Distances)')
    plt.xlabel('Distance to Nearest Known Legitimate Device')
    plt.ylabel('Number of Signals')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('distance_histogram.png')
    print("-> Saved 'distance_histogram.png'")

    print("Generating t-SNE Scatter Plot (This takes a few seconds)...")
    all_features = torch.cat((legit_features, rogue_features), dim=0).cpu().numpy()
    
    legit_labels = legit_y.cpu().numpy()
    rogue_labels = np.full((rogue_features.shape[0],), 99) 
    all_labels = np.concatenate((legit_labels, rogue_labels))

    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    features_2d = tsne.fit_transform(all_features)

    plt.figure(figsize=(12, 8))
    
    scatter = plt.scatter(features_2d[:100, 0], features_2d[:100, 1], 
                          c=legit_labels, cmap='viridis', marker='o', s=50, alpha=0.8, label='Known Devices (1-30)')
    
    plt.scatter(features_2d[100:, 0], features_2d[100:, 1], 
                c='red', marker='X', s=100, label='Spoofed Rogue Signals')

    plt.title('t-SNE Visualization of ResNet Hardware Fingerprints')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('tsne_features.png')
    print("-> Saved 'tsne_features.png'")

if __name__ == "__main__":
    generate_plots()