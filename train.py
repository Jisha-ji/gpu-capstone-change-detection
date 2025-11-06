import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import cv2
import numpy as np
from tqdm import tqdm
from models.unet import UNet
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt

# Dataset class for loading images and masks
class SegmentationDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.images = [f for f in os.listdir(data_dir) if f.endswith('.jpg') or f.endswith('.png')]
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.data_dir, self.images[idx])
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB

        # Create dummy mask if no mask file available (all zeros)
        mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.float32)

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        return image, mask.unsqueeze(0)  # Add channel dimension to mask

# Training function
def train_model(data_dir, num_epochs=5, batch_size=2, lr=0.001, device='cuda'):
    # Data transformations: resize, normalize, and convert to tensor
    transform = A.Compose([
        A.Resize(256, 256),          # Resize all images and masks to 256x256
        A.Normalize(),               # Normalize values to mean=0, std=1
        ToTensorV2()                 # Convert to PyTorch tensors
    ])

    dataset = SegmentationDataset(data_dir=data_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Initialize model, loss function, and optimizer
    model = UNet().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    losses = []

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0

        for imgs, masks in tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            imgs, masks = imgs.to(device), masks.to(device)
            preds = model(imgs)
            loss = criterion(preds, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(dataloader)
        losses.append(avg_loss)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

    # Save model and plot loss
    os.makedirs("artifacts", exist_ok=True)
    torch.save(model.state_dict(), "artifacts/unet_simple.pth")

    plt.plot(losses)
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig("artifacts/training_loss.png")
    plt.close()

    print("Training complete! Model and loss plot saved in 'artifacts/'.")


if __name__ == "__main__":
    data_dir = "data/"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_model(data_dir=data_dir, num_epochs=5, batch_size=2, lr=0.001, device=device)
