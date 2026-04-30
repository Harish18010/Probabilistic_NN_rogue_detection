import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import LoRaDataset
from models.probabilistic import ProbabilisticMultiTaskResNet
from tqdm import tqdm
import os
import sys

def fgsm_attack(signal, epsilon, data_grad):
    """
    The Fast Gradient Sign Method (FGSM).
    Extracts the gradient of the signal and applies mathematical noise 
    in the exact direction that maximizes the model's error.
    """
    sign_data_grad = data_grad.sign()
    perturbed_signal = signal + epsilon * sign_data_grad
    return perturbed_signal

def train_adversarial():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- INITIATING ADVERSARIAL TRAINING (FGSM) ON {str(device).upper()} ---")

    # Load the training data (Subset of 3000 to keep it fast for your CPU)
    train_dataset = LoRaDataset(r'data\raw\Train\dataset_training_no_aug.h5', subset_size=3000)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    # Load your existing 30% dropout model
    model = ProbabilisticMultiTaskResNet(num_devices=30).to(device)
    model.load_state_dict(torch.load("lora_model_drop30.pth", map_location=device, weights_only=True))
    
    criterion = nn.CrossEntropyLoss()
    # Lower learning rate because we are fine-tuning an already trained model
    optimizer = optim.Adam(model.parameters(), lr=0.0001) 

    num_epochs = 5
    epsilon = 0.05 # The intensity of the adversarial hacker noise

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct_clean = 0
        correct_adv = 0
        total = 0

        progress_bar = tqdm(train_loader, desc=f"Adv. Epoch {epoch+1}/{num_epochs}")

        for x, y_dev, _ in progress_bar:
            x, y_dev = x.to(device), y_dev.to(device)
            
            # 1. We must explicitly tell PyTorch to track gradients on the INPUT signal
            x.requires_grad = True

            # 2. Standard Forward Pass (Clean Signal)
            model.zero_grad()
            out_clean, _ = model(x)
            loss_clean = criterion(out_clean, y_dev)
            
            # Calculate gradients to figure out the model's weak spots
            loss_clean.backward(retain_graph=True)
            data_grad = x.grad.data

            # 3. GENERATE THE ADVERSARIAL ATTACK (The Hacker)
            x_adv = fgsm_attack(x, epsilon, data_grad)

            # 4. Forward Pass on the Attacked Signal
            out_adv, _ = model(x_adv)
            loss_adv = criterion(out_adv, y_dev)

            # 5. The Defense: Train the model on BOTH the clean and attacked signals
            total_loss = loss_clean + loss_adv
            
            # We already backwarded clean, now we backward the adversarial part
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            running_loss += total_loss.item()
            
            # Track Accuracies
            _, pred_clean = torch.max(out_clean.data, 1)
            _, pred_adv = torch.max(out_adv.data, 1)
            total += y_dev.size(0)
            correct_clean += (pred_clean == y_dev).sum().item()
            correct_adv += (pred_adv == y_dev).sum().item()

            progress_bar.set_postfix({
                'Loss': f"{running_loss/len(train_loader):.4f}", 
                'Clean_Acc': f"{100.*correct_clean/total:.1f}%",
                'Adv_Acc': f"{100.*correct_adv/total:.1f}%"
            })

    # Save the ultimate, hardened model
    torch.save(model.state_dict(), "lora_model_hardened.pth")
    print("\nAdversarial Training Complete. Model saved to lora_model_hardened.pth")

if __name__ == "__main__":
    train_adversarial()