import os
import torch
import torch.optim as optim
import gc
from datetime import datetime
import argparse
import itertools

from create_dataset import generate_dataset
from supervised_models import generate_resnet

parser = argparse.ArgumentParser()
# Add parameters to the parser
parser.add_argument('--metric', type=str, required=True, help='metric to be estimated: density, mhi, or ed')
parser.add_argument('--imagetype', type=str, required=True, help='data method to use: patches or resized')
parser.add_argument('--trainall', type=bool, default=False, help='train all layers or fully-connected only')
parser.add_argument('--loadmodel', type=str, default=None, help='final path directory and pt of model to load and continue training, e.g., 2023-05-25_23-28-34/11_21753.pt')
parser.add_argument('--newwidth', type=int, default=None, help='image width, if resizing')
parser.add_argument('--newheight', type=int, default=None, help='image height, if resizing')
parser.add_argument('--batchsize', type=int, default=16, help='training batch size')

# Parse and access arguments
args = parser.parse_args()
metric, image_type, train_all, load_model, new_width, new_height, batch_size = args.metric, args.imagetype, args.trainall, args.loadmodel, args.newwidth, args.newheight, args.batchsize

print(f'GPU available: {torch.cuda.is_available()}')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()
gc.collect()

## Load data
train_dataset, val_dataset, test_dataset, train_loader, val_loader, test_loader = generate_dataset(metric, image_type, batch_size, new_width, new_height)

# Define training process
def train():

    # Initialize variables for tracking best validation accuracy
    best_val_loss = float('inf')
    no_improvement_counter = 0

    while True:
        epoch = next(epoch_iterator)
        if epoch > 500:
            break

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
                best_model_path = f'{model_path}/{epoch}_{val_loss:.{deci}f}.pt'
                torch.save({'model_state_dict': model.state_dict(), 'history': history}, best_model_path)
            else:
                no_improvement_counter += 1

            # Early stopping if no improvement after patience epochs    
            if no_improvement_counter >= patience:
                print(f"No improvement in validation loss after {patience} epochs. Training stopped.")
                break

        print(f"Epoch {epoch}: Train Loss: {train_loss:.{deci}f}, Val Loss: {val_loss:.{deci}f}")


# Create model directory
current_datetime = datetime.now()
model_path = f'../saved_models/{metric}/{current_datetime.strftime("%Y-%m-%d_%H-%M-%S")}'
print(model_path)
os.makedirs(model_path)

# Specify patience (number of epochs after which training stops if no validation loss improvement)
patience = 5

# Set display options
if metric == 'density':
    deci = 1
if metric == 'mhi':
    deci = 0
if metric=='ed':
    deci = 2

# Generate model
model, criterion = generate_resnet(device)

# Load previous model, if desired, and initialize history and epoch
if load_model != None:
    checkpoint = torch.load(f'../saved_models/{metric}/{load_model}')
    model.load_state_dict(checkpoint['model_state_dict'])
    history = checkpoint['history']
    train_losses = history['train_losses']
    val_losses = history['val_losses']
    epoch_iterator = itertools.count(len(val_losses)+1)
else:
    train_losses = []
    val_losses = []
    epoch_iterator = itertools.count(1)

# Define optimizer
learning_rate = 0.001
optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

# Train FC layers only, if desired
if train_all == False:
    print('Training FC layers only...')
    train()

# Train all layers, if desired
if train_all == True:

    # Unfreeze the base model parameters
    for param in model.parameters():
        param.requires_grad = True

    # Update optimizer to include all parameters
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

    # Train all layers
    print('Training all layers...')
    train()
