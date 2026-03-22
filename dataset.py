import torch
import os
from torchvision.datasets import OxfordIIITPet
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from collections import Counter

if not os.path.exists('plots'):
    os.mkdir('plots')

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])


train_data = OxfordIIITPet(root='./data', split='trainval', download=True, transform=transform)
test_data  = OxfordIIITPet(root='./data', split='test',     download=True, transform=transform)

print(f"Train samples: {len(train_data)}")
print(f"Test samples:  {len(test_data)}")
print(f"Classes:       {len(train_data.classes)}")

# Class distribution
labels = [train_data[i][1] for i in range(len(train_data))]
counts = Counter(labels)
values = [counts[i] for i in range(len(train_data.classes))]

plt.figure(figsize=(16, 4))
plt.bar(train_data.classes, values)
plt.xticks(rotation=90, fontsize=7)
plt.title("Samples per class")
plt.tight_layout()
plt.savefig("./plots/class_distribution.png")
plt.show()
