import os
os.environ["QT_QPA_PLATFORM"] = "offscreen"
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset import SaliconDataset
from model import SaliencyModel
from torch import nn, optim
from tqdm import tqdm
from torchvision.utils import save_image
from skimage.metrics import structural_similarity as ssim
import numpy as np
import matplotlib.pyplot as plt

# ==== Config ====
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 16
EPOCHS = 15
MODEL_PATH = "model.pth"
IMAGE_ROOT = "SALICON/images"
FIXATION_ROOT = "SALICON/fixations"

# ==== Transforms ====
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# ==== Dataset Loaders ====
def get_loader(split):
    dataset = SaliconDataset(IMAGE_ROOT, FIXATION_ROOT, split, transform=transform)
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=(split == "train"))

train_loader = get_loader("train")
val_loader = get_loader("val")
test_loader = get_loader("test")

# ==== Model, Loss, Optimizer ====
model = SaliencyModel().to(DEVICE)
criterion = nn.BCELoss()  # Model includes Sigmoid
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# ==== Logging ====
loss_history = []
pred_max_history = []
pred_min_history = []

# ==== Train or Load ====
if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH))
    print("✅ Loaded saved model.")
else:
    print("🚀 Training model...")
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        pred_min, pred_max = [], []

        loop = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS}", leave=False)
        for imgs, maps in loop:
            imgs, maps = imgs.to(DEVICE), maps.to(DEVICE)
            preds = model(imgs)

            loss = criterion(preds, maps)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            pred_min.append(preds.min().item())
            pred_max.append(preds.max().item())
            loop.set_postfix(loss=loss.item())

        epoch_loss = running_loss / len(train_loader)
        loss_history.append(epoch_loss)
        pred_min_history.append(np.mean(pred_min))
        pred_max_history.append(np.mean(pred_max))

        print(f"Epoch {epoch + 1}: Loss = {epoch_loss:.4f}")
        print(f"Pred range: {min(pred_min):.2e} → {max(pred_max):.2f}")

    torch.save(model.state_dict(), MODEL_PATH)
    print("💾 Model saved.")

    # ==== Plot Training Curves ====
    os.makedirs("plots", exist_ok=True)

    plt.figure()
    plt.plot(loss_history, label="Train Loss")
    plt.title("Training Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid()
    plt.savefig("plots/loss_curve.png")

    plt.figure()
    plt.plot(pred_min_history, label="Pred Min")
    plt.plot(pred_max_history, label="Pred Max")
    plt.title("Prediction Value Range")
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.legend()
    plt.grid()
    plt.savefig("plots/pred_range.png")

# ==== Evaluation Function ====
@torch.no_grad()
def evaluate(split_name, loader, save_examples=False):
    model.eval()
    total_mse, total_ssim, total_cc, total_kl = 0, 0, 0, 0
    os.makedirs(f"results/{split_name}", exist_ok=True)

    saved = 0
    for i, (imgs, maps) in enumerate(tqdm(loader, desc=f"Evaluating {split_name}", leave=False)):
        imgs, maps = imgs.to(DEVICE), maps.to(DEVICE)
        preds = model(imgs)

        preds_norm = preds
        maps_norm = (maps - maps.min()) / (maps.max() - maps.min() + 1e-8)

        total_mse += nn.functional.mse_loss(preds_norm, maps_norm).item()

        p = torch.nn.functional.log_softmax(preds.view(preds.size(0), -1), dim=1)
        q = torch.nn.functional.softmax(maps.view(maps.size(0), -1), dim=1)
        total_kl += nn.KLDivLoss(reduction="batchmean")(p, q).item()

        pred_np = preds_norm[0, 0].cpu().numpy()
        map_np = maps_norm[0, 0].cpu().numpy()

        ssim_val = ssim(pred_np, map_np, data_range=1.0)
        cc_val = np.corrcoef(pred_np.flatten(), map_np.flatten())[0, 1] if np.std(pred_np) > 0 and np.std(map_np) > 0 else 0.0

        total_ssim += ssim_val
        total_cc += cc_val

        if save_examples and saved < 5:
            for j in range(min(imgs.size(0), 5 - saved)):
                combined = torch.cat([
                    imgs[j],
                    preds_norm[j].repeat(3, 1, 1),
                    maps_norm[j].repeat(3, 1, 1)
                ], dim=2)
                save_image(combined, f"results/{split_name}/example_{saved}.png")
                saved += 1

    n = len(loader)
    print(f"📊 {split_name.capitalize()} Results:")
    print(f"   MSE: {total_mse/n:.4f}")
    print(f"   KL Divergence: {total_kl/n:.4f}")
    print(f"   SSIM: {total_ssim/n:.4f}")
    print(f"   Pearson Corr: {total_cc/n:.4f}")

# ==== Evaluate & Check ====
evaluate("val", val_loader, save_examples=True)
evaluate("test", test_loader, save_examples=True)

print("\n🔍 Sanity check:")
for imgs, maps in train_loader:
    print("Image batch shape:", imgs.shape)
    print("Map batch shape:", maps.shape)
    preds = model(imgs.to(DEVICE))
    print("Predicted shape:", preds.shape)
    break
