import os
import torch
import torchvision.transforms as transforms
from torchvision.io import read_image
import matplotlib.pyplot as plt
from models.unet import UNet

def load_model(model_path, device):
    model = UNet(in_channels=3, out_channels=1).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

def run_inference(model, image_path, device):
    image = read_image(image_path).float() / 255.0
    image = transforms.Resize((128, 128))(image)
    image = image.unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image)
        output = torch.sigmoid(output).cpu().squeeze().numpy()

    return output

def save_output(mask, output_path):
    plt.imshow(mask, cmap='gray')
    plt.axis('off')
    plt.savefig(output_path, bbox_inches='tight')
    print(f"✅ Saved output mask at: {output_path}")

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_path = "artifacts/unet_simple.pth"
    input_image = "data/sample.jpg"   # change to any filename inside data/
    output_image = "artifacts/output_mask.png"

    if not os.path.exists(model_path):
        print("❌ Model not found! Run train.py first.")
    else:
        model = load_model(model_path, device)
        mask = run_inference(model, input_image, device)
        save_output(mask, output_image)

