import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.io import read_image
import matplotlib.pyplot as plt
from tqdm import tqdm
from models.unet import UNet

# Dataset class that loads images and creates fake masks
class SimpleImageDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.img_files = [f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.png'))]
    
    def __len__(self):
        return len(self.img_files)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_files[idx])
        image = read_image(img_path).float() / 255.0
        
        # Auto-generate fake masks (just grayscale threshold images)
        mask = (image.mean(dim=0, keepdim=True) > 0.5).float()
        
        if self.transform:
            image = self.transform(image)
        
        return image, mask

# Training function
def train_model(data_dir, num_epochs=5, batch_size=2, lr=0.001, device='cuda'):
    # Prepare dataset and dataloader
    transform = transforms.Resize((128, 128))
    dataset = SimpleImageDataset(data_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Model initialization
    model = UNet(in_channels=3, out_channels=1).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Training Loop
    model.train()
    losses = []
    for epoch in range(num_epochs):
        epoch_loss = 0
        for imgs, masks in tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            imgs = imgs.to(device)
            masks = masks.to(device)

            outputs = model(imgs)
            loss = criterion(outputs, masks)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
        
        epoch_loss /= len(dataloader)
        losses.append(epoch_loss)
        print(f"Epoch {epoch+1} Loss: {epoch_loss:.4f}")

    # Save model
    os.makedirs('artifacts', exist_ok=True)
    torch.save(model.state_dict(), 'artifacts/unet_simple.pth')

    # Save loss plot
    plt.plot(range(1, num_epochs+1), losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.savefig('artifacts/training_loss.png')
    print("Training complete. Model and loss plot saved in 'artifacts/' folder.")

if __name__ == "__main__":
    data_dir = "data"  # Folder where your images are stored
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_model(data_dir=data_dir, num_epochs=5, batch_size=2, lr=0.001, device=device)

