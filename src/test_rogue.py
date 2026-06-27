import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataset import LoRaDataset
from models.probabilistic import ProbabilisticMultiTaskResNet, enable_dropout
import os
import sys


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import MC_DROPOUT_PASSES, UNCERTAINTY_THRESHOLD

def test_rogue_signals():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- INITIATING ROGUE SIGNAL TEST ON {str(device).upper()} ---")
    
    rogue_file_path = r'data\raw\Test\dataset_rogue.h5'
    
    if not os.path.exists(rogue_file_path):
        print(f"Error: Could not find {rogue_file_path}. Please check your folder path!")
        return

    
    rogue_dataset = LoRaDataset(rogue_file_path, subset_size=10, is_rogue_dataset=True)
    rogue_loader = DataLoader(rogue_dataset, batch_size=10, shuffle=True)
    
    model = ProbabilisticMultiTaskResNet(num_devices=30).to(device)
    
    model.load_state_dict(torch.load("lora_model_drop30.pth", map_location=device, weights_only=True))
    
    model.eval()
    enable_dropout(model) 
    
    
    T = MC_DROPOUT_PASSES 
    uncertainty_threshold = UNCERTAINTY_THRESHOLD 

    for x, y_dev, _ in rogue_loader:
        x = x.to(device)
        mc_predictions = []
        
        for _ in range(T):
            with torch.no_grad():
                out_dev, _ = model(x)
                probs = F.softmax(out_dev, dim=1)
                mc_predictions.append(probs.unsqueeze(0))
                
        mc_predictions = torch.cat(mc_predictions, dim=0)
        
        mean_probs = mc_predictions.mean(dim=0)
        variance = mc_predictions.var(dim=0)
        max_variance = variance.max(dim=1)[0]
        
        print("\n--- ZERO-DAY THREAT DETECTION REPORT ---")
        for i in range(len(x)):
            var = max_variance[i].item()
            
            if var > uncertainty_threshold:
                status = f"SECURITY ALERT: High Uncertainty (Spoof Detected!)"
            else:
                status = "Legitimate Signal"
                
            print(f"Rogue Sample {i+1} | Variance: {var:.6f} | {status}")
            
        break 

if __name__ == "__main__":
    test_rogue_signals()