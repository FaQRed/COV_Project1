import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import OxfordIIITPet
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import os
from model import SimpleCNN, ExperimentCNN1, ExperimentCNN2, ExperimentCNN4, ExperimentCNN5
from sklearn.metrics import classification_report

# CHOOSE EXPERIMENT HERE
# 0 = Baseline
# 1 = BatchNorm + Deeper network
# 2 = Dropout + Augmentation
# 3 = Exp1 + Augmentation + SGD
# 4 = Deep BN + GAP + Dropout
# 5 = GeLU + AdamW + Cosine LR + 224x224

EXPERIMENT = 5

configs = {
    0: {
        "name": "Baseline CNN",
        "model": SimpleCNN(num_classes=37),
        "optimizer": "adam",
        "lr": 0.001,
        "augment": False,
    },
    1: {
        "name": "Exp1 — BatchNorm + Deeper network",
        "model": ExperimentCNN1(num_classes=37),
        "optimizer": "adam",
        "lr": 0.001,
        "augment": False,
    },
    2: {
        "name": "Exp2 — Dropout + Augmentation",
        "model": ExperimentCNN2(num_classes=37),
        "optimizer": "adam",
        "lr": 0.001,
        "augment": True,
    },
    3: {
        "name": "Exp3 — Exp1 + Augmentation + SGD",
        "model": ExperimentCNN1(num_classes=37),
        "optimizer": "sgd",
        "lr": 0.01,
        "augment": True,
    },
    4: {
        "name": "Exp4 — Deep BN + GAP + Dropout",
        "filename": "exp4_deep_bn_gap",
        "model": ExperimentCNN4(num_classes=37),
        "optimizer": "adam",
        "lr": 0.001,
        "augment": False,
    },
    5: {
        "name": "Exp5 — GeLU + AdamW + Cosine LR",
        "model": ExperimentCNN5(num_classes=37),
        "optimizer": "adamw",
        "lr": 0.001,
        "weight_decay": 1e-4,
        "augment": True,
        "scheduler": "cosine",
        "label_smoothing": 0.1,
    },
}

config = configs[EXPERIMENT]
name = config["name"].replace(" ", "_")
print(f"Running: {config['name']}")

if torch.backends.mps.is_available():
    DEVICE = "mps"
elif torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"
print(f"Device: {DEVICE}")

transform_normal = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

transform_augmented = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
])

train_transform = transform_augmented if config["augment"] else transform_normal
test_transform = transform_normal

train_data = OxfordIIITPet(root='./data', split='trainval', download=False, transform=train_transform)
test_data = OxfordIIITPet(root='./data', split='test', download=False, transform=test_transform)

n_val = int(len(train_data) * 0.15)
n_train = len(train_data) - n_val
train_data, val_data = random_split(train_data, [n_train, n_val])

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
val_loader = DataLoader(val_data, batch_size=32)
test_loader = DataLoader(test_data, batch_size=32)

print(f"Train: {len(train_data)} | Val: {len(val_data)} | Test: {len(test_data)}")

model = config["model"].to(DEVICE)
criterion = nn.CrossEntropyLoss(label_smoothing=config.get("label_smoothing", 0.0))

weight_decay = config.get("weight_decay", 0.0)
if config["optimizer"] == "adam":
    optimizer = optim.Adam(model.parameters(), lr=config["lr"], weight_decay=weight_decay)
elif config["optimizer"] == "adamw":
    optimizer = optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=weight_decay)
else:
    optimizer = optim.SGD(model.parameters(), lr=config["lr"], momentum=0.9, weight_decay=weight_decay)

EPOCHS = 50
patience = 10
no_improve = 0
best_val_acc = 0

train_acc_history = []
val_acc_history = []

os.makedirs("plots", exist_ok=True)
os.makedirs("models", exist_ok=True)

scheduler = None
if config.get("scheduler") == "cosine":
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

for epoch in range(1, EPOCHS + 1):

    # Train
    model.train()
    correct, total = 0, 0

    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        correct += (outputs.argmax(1) == labels).sum().item()
        total += labels.size(0)

    train_acc = correct / total

    # Validation
    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)

    val_acc = correct / total

    train_acc_history.append(train_acc)
    val_acc_history.append(val_acc)

    current_lr = optimizer.param_groups[0]["lr"]
    print(f"Epoch {epoch}/{EPOCHS} | Train: {train_acc:.3f} | Val: {val_acc:.3f} | LR: {current_lr:.6f}")

    if scheduler is not None:
        scheduler.step()

    # Early stopping
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        no_improve = 0
        torch.save(model.state_dict(), f"models/best_model_exp{EXPERIMENT}.pth")
    else:
        no_improve += 1
        if no_improve >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

# Test
model.load_state_dict(torch.load(f"models/best_model_exp{EXPERIMENT}.pth", weights_only=True))
model.eval()
correct, total = 0, 0
all_preds  = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        outputs = model(images)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += labels.size(0)
        all_preds.extend(outputs.argmax(1).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

print(f"\nFinal test accuracy: {correct / total:.3f}")

# Plotting results
plt.figure(figsize=(8, 5))
plt.plot(train_acc_history, label='Train accuracy')
plt.plot(val_acc_history, label='Val accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title(config["name"])
plt.legend()
plt.grid()
plt.savefig(f"plots/learning_curve_exp_{EXPERIMENT}_{name}.png")
plt.show()

# Print report
class_names = test_data.classes
report = classification_report(all_labels, all_preds, target_names=class_names)
print(report)

# Save to file
with open("results.txt", "a") as f:
    f.write(f"Experiment {EXPERIMENT}: {config['name']}\n")
    f.write(f"  Test accuracy: {correct / total:.3f}\n")
    f.write(f"  Best val acc:  {best_val_acc:.3f}\n")
    f.write(f"  Epochs run:    {len(train_acc_history)}\n")
    f.write("\nClassification Report:\n")
    f.write(report)
    f.write("-" * 40 + "\n")