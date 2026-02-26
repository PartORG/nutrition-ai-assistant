from torchvision import datasets, transforms

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

train_dataset = datasets.Food101(
    root="./data",
    split="train",
    download=True,
    transform=transform
)

val_dataset = datasets.Food101(
    root="./data",
    split="test",
    download=True,
    transform=transform
)

print("Food101 dataset downloaded successfully.")