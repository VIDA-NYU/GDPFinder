
import os
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.transforms import transforms
import random
import gc
from datetime import datetime

Image.MAX_IMAGE_PIXELS = None

print(torch.cuda.is_available())
print(torch.version.cuda)


## Create datasets

class CustomDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.image_files = os.listdir(data_dir)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        image_name = self.image_files[index]
        image_path = os.path.join(self.data_dir, image_name)

        # Load the image
        image = Image.open(image_path)

        # Extract the label from the image name
        label = int(image_name.split('_')[-2].split('.')[0])

        # Apply transformations if provided
        if self.transform:
            image = self.transform(image)

        return image, label

# Define the transformations to apply to the images
transform = transforms.Compose([
    transforms.ToTensor()
])

# Define the path to the directory containing the image files
data_dir = '../data/patches'

# Set a random seed for reproducibility
random.seed(42)

# Create an instance of the custom dataset
dataset = CustomDataset(data_dir, transform=transform)

# Calculate the sizes for training, validation, and testing sets
dataset_size = len(dataset)
train_size = int(0.7 * dataset_size)
val_size = int(0.15 * dataset_size)
test_size = dataset_size - train_size - val_size

# Create random indices for splitting the dataset
indices = list(range(dataset_size))
random.shuffle(indices)

# Split the dataset into training, validation, and testing sets
train_indices = indices[:train_size]
val_indices = indices[train_size:train_size + val_size]
test_indices = indices[train_size + val_size:]

# Create the training dataset
train_dataset = torch.utils.data.Subset(dataset, train_indices)

# Create the validation dataset
val_dataset = torch.utils.data.Subset(dataset, val_indices)

# Create the testing dataset
test_dataset = torch.utils.data.Subset(dataset, test_indices)

print("Training set size:", len(train_dataset))
print("Validation set size:", len(val_dataset))
print("Testing set size:", len(test_dataset))




## Generate ResNet

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Clear GPU cache
torch.cuda.empty_cache()
gc.collect()

# Define ResNet-50 model
model = resnet50(weights=ResNet50_Weights.DEFAULT)
num_features = model.fc.in_features

# Freeze the base model parameters
for param in model.parameters():
    param.requires_grad = False

# Add additional fully connected layers
model.fc = nn.Sequential(
    nn.Linear(num_features, 1024),
    nn.ReLU(inplace=True),
    nn.Linear(1024, 224),
    nn.ReLU(inplace=True),
    nn.Linear(224, 64),
    nn.ReLU(inplace=True),
    nn.Linear(64, 1)
)
model.to(device)

# Define loss function
criterion = nn.L1Loss()

# Define optimizer
optimizer = optim.AdamW(model.parameters(), lr=0.01)

# Create data loaders
batch_size = 8
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size)



## Train model

# Create model directory
current_datetime = datetime.now()
model_dir = f'../saved_models/individual_patches/{current_datetime.strftime("%Y-%m-%d_%H-%M-%S")}'
os.makedirs(model_dir)

num_epochs = 500

# Train top layers
# Initialize variables for tracking best validation accuracy
best_val_loss = float('inf')

# Initialize lists to store historical training and validation loss
train_losses = []
val_losses = []

for epoch in range(num_epochs):
    # Training
    model.train()
    train_loss = 0.0
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images).squeeze(1)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        train_losses.append(train_loss) # for history

    train_loss /= len(train_loader)

    # Validation
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images).squeeze(1)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

        val_loss /= len(val_loader)
        val_losses.append(val_loss) # for history

        # Check for improvement in validation accuracy
        if abs(val_loss) < best_val_loss:
            best_val_loss = abs(val_loss)
            history = {'train_losses': train_losses, 'val_losses': val_losses}
            best_model_path = f'{model_dir}/FC_{epoch+1}_{val_loss:.0f}.pt'
            torch.save({'model_state_dict': model.state_dict(), 'history': history}, best_model_path)
        else:
            print(f"No improvement in validation loss. FC layer training stopped.")
            break

    print(f"Training FC layers. Epoch {epoch + 1}/{num_epochs}: Train Loss: {train_loss:.0f}, Val Loss: {val_loss:.0f}")



# Unfreeze the base model parameters
for param in model.parameters():
    param.requires_grad = True

# Update optimizer to include all parameters
optimizer = optim.AdamW(model.parameters(), lr=0.01)

# Train all layers
patience = 5
no_improvement_counter = 0

# Initialize variables for tracking best validation accuracy
best_val_loss = float('inf')

for epoch in range(num_epochs):
    # Training
    model.train()
    train_loss = 0.0
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images).squeeze(1)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(train_loader)
    train_losses.append(train_loss) # for history

    # Validation
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images).squeeze(1)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

        val_loss /= len(val_loader)
        val_losses.append(val_loss) # for history

        # Check for improvement in validation accuracy
        if abs(val_loss) < best_val_loss:
            best_val_loss = abs(val_loss)
            no_improvement_counter = 0
            history = {'train_losses': train_losses, 'val_losses': val_losses}
            best_model_path = f'{model_dir}/{epoch+1}_{val_loss:.0f}.pt'
            torch.save({'model_state_dict': model.state_dict(), 'history': history}, best_model_path)
        else:
            no_improvement_counter += 1

        # Early stopping if no improvement after patience epochs
        if no_improvement_counter >= patience:
            print(f"No improvement in validation loss after {patience} epochs. Training stopped.")
            break

    print(f"Epoch {epoch + 1}/{num_epochs}: Train Loss: {train_loss:.0f}, Val Loss: {val_loss:.0f}")