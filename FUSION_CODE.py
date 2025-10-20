import os
import pandas as pd
import shutil

# File and folder paths
oct_csv = '/content/OCT_csv/OCT.csv'
fundus_csv = '/content/FUNDUS_csv/EYE FUNDUS.csv'
oct_img_dir = '/content/OCT_image'
fundus_img_dir = '/content/FUNDUS_image'
output_dir = '/content/OUTPUT'

# Load CSVs
oct_df = pd.read_csv(oct_csv)
fundus_df = pd.read_csv(fundus_csv)

def get_patient_id(image_name):
    return image_name.split('_')[0]

# Create patient DR status dict
patient_status = {}

# Helper to check if DR label means diabetic retinopathy
def is_dr_label(label):
    if isinstance(label, str):
        label = label.upper().strip()
        return label in ['NPDR', 'PDR']
    return label != 0

# Mark patient based on OCT images
for _, row in oct_df.iterrows():
    pid = get_patient_id(row['Name'])
    if pid not in patient_status:
        patient_status[pid] = False
    if is_dr_label(row['DR']):
        patient_status[pid] = True

# Mark patient based on Fundus images
for _, row in fundus_df.iterrows():
    pid = get_patient_id(row['Name'])
    if pid not in patient_status:
        patient_status[pid] = False
    if is_dr_label(row['DR']):
        patient_status[pid] = True

# Organize images
os.makedirs(output_dir, exist_ok=True)

for pid, has_dr in patient_status.items():
    folder_name = 'DR' if has_dr else 'NORMAL'
    patient_folder = os.path.join(output_dir, folder_name, f'Patient_{pid}')
    os.makedirs(patient_folder, exist_ok=True)

    # Copy OCT images
    for img_name in oct_df[oct_df['Name'].str.startswith(pid)]['Name']:
        src = os.path.join(oct_img_dir, img_name + '.jpg')
        dst = os.path.join(patient_folder, img_name + '.jpg')
        if os.path.exists(src):
            shutil.copy(src, dst)

    # Copy Fundus images
    for img_name in fundus_df[fundus_df['Name'].str.startswith(pid)]['Name']:
        src = os.path.join(fundus_img_dir, img_name + '.jpg')
        dst = os.path.join(patient_folder, img_name + '.jpg')
        if os.path.exists(src):
            shutil.copy(src, dst)



import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
import os

class OCTFundusDataset(Dataset):
    def __init__(self, oct_dir, fundus_dir, oct_csv, fundus_csv, transform=None):
        self.oct_dir = oct_dir
        self.fundus_dir = fundus_dir
        self.transform = transform

        # Load CSVs
        self.oct_df = pd.read_csv(oct_csv)
        self.fundus_df = pd.read_csv(fundus_csv)

        # Map image name to label for OCT
        self.oct_labels = dict(zip(self.oct_df['Name'], self.oct_df['DR']))

        # Map image name to label for fundus
        self.fundus_labels = dict(zip(self.fundus_df['Name'], self.fundus_df['DR']))

        # Find patient IDs (assumes same patients across modalities)
        self.patients = sorted(set(self.oct_df['Name'].apply(lambda x: x.split('_')[0])))

        # Create list of paired samples (tuple: (oct_img_name, fundus_img_name))
        self.samples = []
        for pid in self.patients:
            oct_imgs = [name for name in self.oct_labels if name.startswith(pid)]
            fundus_imgs = [name for name in self.fundus_labels if name.startswith(pid)]
            for o_img in oct_imgs:
                for f_img in fundus_imgs:
                    # Pair OCT and fundus images from same patient; can customize further
                    self.samples.append((o_img, f_img))

        # Optional: Map string DR labels to integers for classification
        self.label_map = {'0': 0, 'NPDR': 1, 'PDR': 2, '-': -1}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        o_img_name, f_img_name = self.samples[idx]

        # Load images
        o_img_path = os.path.join(self.oct_dir, o_img_name + ".jpg")
        f_img_path = os.path.join(self.fundus_dir, f_img_name + ".jpg")
        o_img = Image.open(o_img_path).convert("RGB")
        f_img = Image.open(f_img_path).convert("RGB")

        # Get labels (using OCT label here, can also combine or check consistency)
        o_label_str = self.oct_labels[o_img_name]
        label = self.label_map.get(str(o_label_str), -1)

        # Apply transforms
        if self.transform:
            o_img = self.transform(o_img)
            f_img = self.transform(f_img)

        return o_img, f_img, label




import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

# Paths to your folders
data_dir = "/content/OUTPUT"  # Parent folder containing 'Normal' and 'DR' subfolders

# Transformations (resize, normalize, augmentation if needed)
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# Load dataset using ImageFolder
dataset = datasets.ImageFolder(root=data_dir, transform=transform)

# Split dataset into train-val-test: e.g. 70%-15%-15%
total_size = len(dataset)
train_size = int(0.7 * total_size)
val_size = int(0.15 * total_size)
test_size = total_size - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

print(f"Training samples: {len(train_dataset)}")
print(f"Validation samples: {len(val_dataset)}")
print(f"Testing samples: {len(test_dataset)}")
print(f"Class names: {dataset.classes}")




import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models

# Device (GPU if available)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load pretrained ResNet18 and modify for 2 classes
model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, 2)  # Output: 2 classes (Normal, DR)
model = model.to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop function
def train(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    accuracy = correct / total
    return epoch_loss, accuracy

# Example usage:
# for epoch in range(num_epochs):
#     train_loss, train_acc = train(model, train_loader, criterion, optimizer, device)
#     print(f'Epoch {epoch+1}, Loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}')





def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    epoch_loss = running_loss / total
    accuracy = correct / total
    return epoch_loss, accuracy





num_epochs = 10

for epoch in range(num_epochs):
    train_loss, train_acc = train(model, train_loader, criterion, optimizer, device)
    val_loss, val_acc = validate(model, val_loader, criterion, device)
    print(f"Epoch {epoch+1}: Train loss {train_loss:.4f}, Train acc {train_acc:.4f}, Val loss {val_loss:.4f}, Val acc {val_acc:.4f}")





def test(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    accuracy = correct / total
    print(f'Test Accuracy: {accuracy:.4f}')
    return accuracy






def evaluate_accuracy(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    accuracy = correct / total
    print(f'Accuracy: {accuracy * 100:.2f}%')
    return accuracy

# Example usage:
accuracy = evaluate_accuracy(model, test_loader, device)
