import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataset import LoRaDataset
from models.probabilistic import ProbabilisticMultiTaskResNet
import os
import sys


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import KNN_ANOMALY_THRESHOLD

def extract_features(model, x):
    x = x.view(-1, 1, 128, 128)
    x = F.relu(model.bn1(model.conv1(x)))
    x = model.layer1(x)
    x = model.layer2(x)
    x = model.layer3(x)
    x = model.attention(x)
    x = model.pool(x)
    features = x.view(x.size(0), -1)
    return features

def test_distance_anomaly():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- INITIATING FEATURE-SPACE ANOMALY DETECTION ON {str(device).upper()} ---")
    
    
    model = ProbabilisticMultiTaskResNet(num_devices=30).to(device)
   
    model.load_state_dict(torch.load("lora_model_drop30.pth", map_location=device, weights_only=True))
    model.eval() 
    
   
    print("Building database from legitimate devices...")
    legit_dataset = LoRaDataset(r'data\raw\Train\dataset_training_no_aug.h5', subset_size=105)
    legit_loader = DataLoader(legit_dataset, batch_size=105, shuffle=True)
    
    
    legit_batch_x, _, _ = next(iter(legit_loader))
    
   
    db_x = legit_batch_x[:100]
    legit_test_x = legit_batch_x[100:] 

    with torch.no_grad():
        legit_database = extract_features(model, db_x.to(device))
        
   
    rogue_file_path = r'data\raw\Test\dataset_rogue.h5'
    if not os.path.exists(rogue_file_path):
        print("Error: Rogue dataset not found!")
        return

    print("\nEvaluating Rogue Signals...")
   
    rogue_dataset = LoRaDataset(rogue_file_path, subset_size=10, is_rogue_dataset=True)
    rogue_loader = DataLoader(rogue_dataset, batch_size=10, shuffle=True)
    rogue_x, _, _ = next(iter(rogue_loader))
    
   
    test_signals = torch.cat((legit_test_x, rogue_x), dim=0).to(device)
    
    with torch.no_grad():
        test_features = extract_features(model, test_signals)
        
        
        distances_matrix = torch.cdist(test_features, legit_database)
        
        
        min_distances, _ = torch.min(distances_matrix, dim=1)
        
    print("\n--- NEAREST NEIGHBOR ANOMALY REPORT ---")
    
   
    for i, dist in enumerate(min_distances):
        val = dist.item()
        true_type = "LEGITIMATE" if i < 5 else "ROGUE"
        
        if val > KNN_ANOMALY_THRESHOLD:
            status = "SECURITY ALERT: Hardware Fingerprint Unrecognized!"
        else:
            status = "Authenticated"
            
        print(f"Sample {i+1} ({true_type}) | Distance: {val:.4f} | {status}")

if __name__ == "__main__":
    test_distance_anomaly()