import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from utils import load_config, set_seed, classification_metrics
from data_pipeline import EmotionDataset, FaceExtractor
from models import get_model

def train_loop(cfg):
    set_seed(cfg['train'].get('seed', 42))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    classes = ["neutral", "happy", "sad", "angry", "surprise", "fear", "disgust", "contempt", "boredom"]
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((cfg['data']['img_size'], cfg['data']['img_size'])),
        transforms.ToTensor(),
        transforms.Normalize(mean=cfg['data']['mean'], std=cfg['data']['std'])
    ])

    train_ds = EmotionDataset("data/train", classes, transform)
    val_ds = EmotionDataset("data/val", classes, transform)
    train_loader = DataLoader(train_ds, batch_size=cfg['train']['batch_size'], shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=cfg['train']['batch_size'], shuffle=False, num_workers=0)

    model = get_model(cfg['model']['name'], num_classes=cfg['model']['num_classes'], pretrained=cfg['model']['pretrained'])
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=float(cfg['train']['lr']), weight_decay=float(cfg['train'].get('weight_decay', 0)))

    best_f1 = 0.0
    os.makedirs("outputs", exist_ok=True)

    for epoch in range(cfg['train']['epochs']):
        model.train()
        running_loss = 0.0
        for imgs, labels in tqdm(train_loader):
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * imgs.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)

        # تقييم
        model.eval()
        y_true, y_pred = [], []
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                preds = torch.argmax(outputs, dim=1)
                y_true.extend(labels.cpu().numpy())
                y_pred.extend(preds.cpu().numpy())

        metrics = classification_metrics(y_true, y_pred)
        print(f"Epoch {epoch+1}: loss={epoch_loss:.4f} acc={metrics['accuracy']:.4f} f1={metrics['f1']:.4f}")

        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            torch.save(model.state_dict(), os.path.join("outputs", "best.pth"))
            print("Saved best model")

    print("Training finished")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    train_loop(cfg)
