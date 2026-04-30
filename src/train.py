import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import LoRaDataset
from models.probabilistic import ProbabilisticMultiTaskResNet
from tqdm import tqdm

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_dataset = LoRaDataset(r'data\raw\Train\dataset_training_no_aug.h5', subset_size=15000)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    model = ProbabilisticMultiTaskResNet(num_devices=30).to(device)

    criterion_device = nn.CrossEntropyLoss()
    criterion_rogue = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 30

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct_device = 0
        correct_rogue = 0
        total = 0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")

        for x, y_dev, y_rogue in progress_bar:
            x, y_dev, y_rogue = x.to(device), y_dev.to(device), y_rogue.to(device)

            optimizer.zero_grad()

            out_dev, out_rogue = model(x)

            loss_dev = criterion_device(out_dev, y_dev)
            loss_rog = criterion_rogue(out_rogue, y_rogue)
            loss = loss_dev + loss_rog

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            
            _, predicted_dev = torch.max(out_dev.data, 1)
            _, predicted_rogue = torch.max(out_rogue.data, 1)
            
            total += y_dev.size(0)
            correct_device += (predicted_dev == y_dev).sum().item()
            correct_rogue += (predicted_rogue == y_rogue).sum().item()

            progress_bar.set_postfix({
                'Loss': f"{running_loss/len(train_loader):.4f}", 
                'Dev_Acc': f"{100.*correct_device/total:.2f}%",
                'Rogue_Acc': f"{100.*correct_rogue/total:.2f}%"
            })

   
  
    torch.save(model.state_dict(), "lora_model_drop30.pth")
    print("\nModel saved to lora_model_drop30.pth")


if __name__ == "__main__":
    train()