import os
import torch
import torch.optim as optim
import gc
from datetime import datetime

from create_dataset import generate_dataset
from supervised_models import generate_resnet

print(f'GPU available: {torch.cuda.is_available()}')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()
gc.collect()

## Load data
train_dataset, val_dataset, test_dataset, train_loader, val_loader, test_loader = generate_dataset()

# Load model architechture
model, criterion = generate_resnet(device)

# Define optimizer
optimizer = optim.AdamW(model.parameters(), lr=0.01)

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
        if val_loss < best_val_loss:
            best_val_loss = val_loss
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
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improvement_counter = 0
            history = {'train_losses': train_losses, 'val_losses': val_losses}
            best_model_path = f'{model_dir}/All_{epoch+1}_{val_loss:.0f}.pt'
            torch.save({'model_state_dict': model.state_dict(), 'history': history}, best_model_path)
        else:
            no_improvement_counter += 1

        # Early stopping if no improvement after patience epochs
        if no_improvement_counter >= patience:
            print(f"No improvement in validation loss after {patience} epochs. Training stopped.")
            break

    print(f"Training entire model. Epoch {epoch + 1}/{num_epochs}: Train Loss: {train_loss:.0f}, Val Loss: {val_loss:.0f}")