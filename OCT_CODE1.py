import pandas as pd
import shutil
import os

# Path to your CSV and images
csv_path = '/content/csv_path/OCT.csv'
img_folder = '/content/img_folder'  # Change to your images folder
output_folder = '/content/output_folder'

# Create output folders if they don't exist
os.makedirs(os.path.join(output_folder, 'DR'), exist_ok=True)
os.makedirs(os.path.join(output_folder, 'Normal'), exist_ok=True)

# Read CSV
df = pd.read_csv(csv_path)

for idx, row in df.iterrows():
    img_name = f"{row['Name']}.{row['Format']}"
    dr_label = str(row['DR']).strip().upper()
    src = os.path.join(img_folder, img_name)
    if dr_label in ['NPDR', 'PDR']:
        dst = os.path.join(output_folder, 'DR', img_name)
    else:
        dst = os.path.join(output_folder, 'Normal', img_name)
    if os.path.exists(src):
        shutil.copy(src, dst)
    else:
        print(f"Image not found: {src}")




from torchvision import datasets, transforms
from torch.utils.data import DataLoader

data_dir = '/content/output_folder'  # Path to your sorted images

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

dataset = datasets.ImageFolder(root=data_dir, transform=transform)
loader = DataLoader(dataset, batch_size=32, shuffle=True)





import torch
import torch.nn as nn
from torchvision import models

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, 2)  # 2 classes: DR and Normal
model = model.to(device)





criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

n_epochs = 10
for epoch in range(n_epochs):
    model.train()
    running_loss, correct = 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
    print(f"Epoch {epoch+1}: Loss {running_loss/len(dataset):.4f}, Accuracy {correct/len(dataset):.4f}")







from PIL import Image
import torchvision.transforms as transforms

img_path = '/content/test/DR4.jpeg'  # Path to your image
image = Image.open(img_path).convert('RGB')

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

input_tensor = transform(image).unsqueeze(0).to(device)








model.eval()
with torch.no_grad():
    output = model(input_tensor)
    _, predicted = torch.max(output, 1)
    class_idx = predicted.item()
    class_names = ['DR ', 'Normal']  # or ['Normal', 'DR'] depending on your folder order
    print(f'This image is predicted as: {class_names[class_idx]}')









model.eval()
correct = 0
total = 0

with torch.no_grad():
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy: {100 * correct / total:.2f}%')
