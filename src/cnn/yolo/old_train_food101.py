# General imports

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, Subset
import random


# 1. Device (CPU only)

device = torch.device("cpu") 


# 2. Transforms

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])


# 3. Load Full Food101 Dataset

full_train_dataset = datasets.Food101(
    root="./data_cnn",
    split="train",
    download=True,
    transform=transform
)


# 4. Create Subset (to train faster on CPU, adjust subset_size as needed)

subset_size = 20000  
indices = random.sample(range(len(full_train_dataset)), subset_size)
train_dataset = Subset(full_train_dataset, indices)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

print(f"Training on subset of {subset_size} images")


# 5. Load Pretrained ResNet18

model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

# Freeze backbone
for param in model.parameters():
    param.requires_grad = False

# Replace final layer (101 classes)
model.fc = nn.Linear(model.fc.in_features, 101)

model = model.to(device)


# 6. Loss and Optimizer

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=0.001)


# 7. Training Loop (CPU Safe)

epochs = 5

for epoch in range(epochs):
    model.train()
    running_loss = 0.0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch {epoch+1}/{epochs} - Loss: {running_loss/len(train_loader):.4f}")

print("Training finished.")


# 8. Save Model

torch.save(model.state_dict(), "food101_model.pth")
print("Model saved as food101_model.pth")













