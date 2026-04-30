import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataset import LoRaDataset
from models.probabilistic import ProbabilisticMultiTaskResNet, enable_dropout

def test_mc_dropout():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    test_dataset = LoRaDataset(r'data\raw\Train\dataset_training_no_aug.h5', subset_size=500)
    test_loader = DataLoader(test_dataset, batch_size=5, shuffle=True)
    
    model = ProbabilisticMultiTaskResNet(num_devices=30).to(device)
    
   
    model.load_state_dict(torch.load("lora_model_drop30.pth", map_location=device, weights_only=True))
    
    model.eval()
    enable_dropout(model) 
    
    T = 10 
    uncertainty_threshold = 0.05

    for x, y_dev, y_rogue in test_loader:
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
        
        print("\n--- MC Dropout Uncertainty Test ---")
        for i in range(len(x)):
            pred_class = mean_probs[i].argmax().item()
            var = max_variance[i].item()
            
            status = "SECURITY ALERT: High Uncertainty" if var > uncertainty_threshold else "Legitimate Signal"
            
            print(f"Sample {i+1} | Pred Device: {pred_class} | Variance: {var:.6f} | {status}")

if __name__ == "__main__":
    test_mc_dropout()