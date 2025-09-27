import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from PIL import Image
from sklearn.model_selection import train_test_split
import pandas as pd

# Обновлённые классы почвы
CLASSES = ['Red', 'Black', 'Desert', 'Alluvial']
NUM_CLASSES = len(CLASSES)

class SatelliteSoilDataset(Dataset):
    """Датасет для спутниковых изображений почвы с обработкой ошибок."""
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        print(f"Проверка папки: {root_dir}")
        for class_idx, class_name in enumerate(CLASSES):
            class_dir = os.path.join(root_dir, class_name)
            print(f"Проверка подкаталога: {class_dir}")
            if os.path.exists(class_dir):
                print(f"Содержимое {class_dir}:")
                for img_name in os.listdir(class_dir):
                    img_path = os.path.join(class_dir, img_name)
                    print(f"  - {img_name}")
                    if img_name.lower().endswith(('.jpg')):
                        try:
                            with Image.open(img_path) as img:
                                img.load()  # Более строгая проверка
                                self.samples.append((img_path, class_idx))
                        except (IOError, OSError) as e:
                            print(f"Пропущен повреждённый файл {img_path}: {str(e)}")
            else:
                print(f"Подкаталог {class_dir} не найден")
        print(f"Найдено {len(self.samples)} образцов в {root_dir}")
        if len(self.samples) == 0:
            raise ValueError(f"Датасет пуст. Проверьте путь {root_dir} и структуру папок.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        print(f"Открытие файла: {img_path}")  # Отладка
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        print(f"Открытие файла: {img_path}")  # Отладка
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label

# Трансформации
transform_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
transform_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class SoilCNN(nn.Module):
    """CNN-модель для классификации почвы по спутниковым снимкам с 4 классами."""
    def __init__(self, num_classes=NUM_CLASSES):
        super(SoilCNN, self).__init__()
        from torchvision.models import ResNet18_Weights
        self.base_model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.base_model.fc = nn.Linear(self.base_model.fc.in_features, num_classes)

    def forward(self, x):
        return self.base_model(x)

def train_model(model, train_loader, val_loader, epochs=10, lr=0.001):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        print(f'Epoch {epoch+1}: Train Loss: {running_loss/len(train_loader):.4f}, Val Loss: {val_loss/len(val_loader):.4f}, Val Acc: {100 * correct / total:.2f}%')
    
    torch.save(model.state_dict(), 'soil_model_4classes.pth')
    return model

if __name__ == '__main__':
    dataset_dir = sys.argv[1] if len(sys.argv) > 1 else 'C:/Users/Zakhar/VSCode/NumericalMethods/TestSoilType/database'
    full_dataset = SatelliteSoilDataset(dataset_dir, transform=transform_train)
    
    # Разделение датасета
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset = SatelliteSoilDataset(dataset_dir, transform=transform_train)
    val_dataset = SatelliteSoilDataset(dataset_dir, transform=transform_test)
    
    train_indices, val_indices = train_test_split(range(len(full_dataset)), train_size=train_size, test_size=val_size, random_state=42)
    train_loader = DataLoader(train_dataset, batch_size=32, sampler=torch.utils.data.SubsetRandomSampler(train_indices), num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=32, sampler=torch.utils.data.SubsetRandomSampler(val_indices), num_workers=0)
    
    model = SoilCNN()
    trained_model = train_model(model, train_loader, val_loader, epochs=10)