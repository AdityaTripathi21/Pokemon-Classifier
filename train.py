import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np 
from sklearn.model_selection import train_test_split
import pandas as pd

from dataset import PokemonDataset
from model import SimpleCNN

batch_size = 32
lr = 0.001
num_epochs = 50
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# train transform - has a lot of random transforms so model doesn't overfit
train_transform = transforms.Compose([  # compose chains transforms sequentially
    transforms.Resize((128, 128)),      # every image is 128 x 128
    transforms.RandomHorizontalFlip(),  # 50% chance of random horizontal flip
    transforms.RandomRotation(15),      # random rotation by 15 degrees
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),   # change color properities randomly
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),   # random translations in x and y direction
    transforms.ToTensor(),              # convert PIL image to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406],    # normalize all channels
                         std=[0.229, 0.224, 0.225])
])

# validation transform, only has necessary transforms because we want validation to be consistent
test_transform = transforms.Compose([
    transforms.Resize((128, 128)), 
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

df = pd.read_csv("data/pokemon.csv")


train_df, test_df = train_test_split(
    df, test_size=0.2, train_size=0.8, 
    random_state=42, stratify=df["Type1"])  # stratify ensures that both train and test data have the same proportion of Type1,
                                            # otherwise some types might be overrepresented in training and some might be under

# reset indices because we split our dataset, so indices aren't sequential 
train_df = train_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)

train_dataset = PokemonDataset(
    img_df=train_df,
    images_dir="data/images",
    transform=train_transform
)

test_dataset = PokemonDataset(
    img_df=test_df,
    images_dir="data/images",
    transform=test_transform
)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)   # shuffle during training so model doesn't learn data
                                                                        # every epoch, model would only see few types and in order

test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)     # don't shuffle validation because it needs to be consistent


num_classes = len(train_dataset.all_types)

model = SimpleCNN(num_classes=num_classes).to(device)

loss_function = nn.BCEWithLogitsLoss()  # see model.py


optimizer = optim.Adam(
    model.parameters(),     # All weights in the model
    lr=lr,                  
    weight_decay=1e-4       # L2 regularization (prevents overfitting), Adam decouples weight decay from gradient
)

def train_epoch(model, dataloader, loss_function, optimizer, device):
    model.train()   # set model to training mode

    running_loss = 0.0
    
    for index, (images, labels) in enumerate(dataloader):

        # Move batch to GPU
        images = images.to(device)  # [32, 3, 128, 128]
        labels = labels.to(device)  # [32, 18]
        
        # Clear old gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(images)  # [32, 18] raw logits
        
        # Calculate loss
        loss = loss_function(outputs, labels)

        # Backward pass (compute gradients)
        loss.backward()
        
        # Update weights
        optimizer.step()
        
        # Track loss
        running_loss += loss.item()

        # track loss over every 5 batches
        if index % 5 == 0:
            print(f"    Batch [{index}/{len(dataloader)}], Loss: {loss.item():.4f}")
    
    # Return average loss for this epoch
    avg_loss = running_loss / len(dataloader)
    return avg_loss

def validate(model, dataloader, loss_function, device):
        model.eval()  # Set to evaluation mode (dropout off)

        running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():  # Don't compute gradients (saves memory)
            for images, labels in dataloader:
                images = images.to(device)
                labels = labels.to(device)
            
                # Forward pass only
                outputs = model(images)
                loss = loss_function(outputs, labels)
                running_loss += loss.item()
            
                # Calculate accuracy
                predictions = (torch.sigmoid(outputs) > 0.3).float()    # output is raw logits. 0.5 is threshold

                correct += (predictions == labels).sum().item()  # predictions == labels creates a True/False tensor
                                                                            # sum counts how many samples in each batch are correct
                                                                            # item converts tensor to scalar

                total += labels.numel()  # Total number of predictions (batch_size * 18)
    
        avg_loss = running_loss / len(dataloader)
        accuracy = correct / total
        return avg_loss, accuracy


train_losses = []
test_losses = []
test_accuracies = []
best_test_loss = float('inf')

for epoch in range(num_epochs):
    print(f"Epoch [{epoch+1}/{num_epochs}]")
    train_loss = train_epoch(model, train_loader, loss_function, optimizer, device)
    test_loss, test_acc = validate(model, test_loader, loss_function, device)

    train_losses.append(train_loss)
    test_losses.append(test_loss)
    test_accuracies.append(test_acc)

    print(f"  Train Loss: {train_loss:.4f}")
    print(f"  Test Loss:  {test_loss:.4f} | Test Accuracy: {test_acc:.4f}")

    if test_loss < best_test_loss:
        best_test_loss = test_loss
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'test_loss': test_loss,
            'test_acc': test_acc,
        }, 'best_pokemon_model.pth')
        print("  ✓ Saved best model")
    
    print()

print(f"Best test loss: {best_test_loss:.4f}")

plt.figure(figsize=(15, 5))

# Plot 1: Loss
plt.subplot(1, 3, 1)
plt.plot(train_losses, label='Train Loss', linewidth=2)
plt.plot(test_losses, label='Test Loss', linewidth=2)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.title('Training and Test Loss', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 2: Accuracy
plt.subplot(1, 3, 2)
plt.plot(test_accuracies, label='Test Accuracy', color='green', linewidth=2)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.title('Test Accuracy', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 3: Overfitting check
plt.subplot(1, 3, 3)
gap = [train_losses[i] - test_losses[i] for i in range(len(train_losses))]
plt.plot(gap, label='Train-Test Gap', color='red', linewidth=2)
plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Loss Gap', fontsize=12)
plt.title('Overfitting Check (smaller is better)', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('training_results.png', dpi=150)
plt.show()

print("\nTraining curves saved to 'training_results.png'")



        



