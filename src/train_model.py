import sys
import os

# Add project root to sys.path to allow imports from src
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import numpy as np
import time

from src.neural_network.dataset import ECGDataset, RandomNoise, RandomShift, Compose
from src.neural_network.model import CNNLSTM

def train():
    # Hyperparameters
    BATCH_SIZE = 32 # Increased for stability
    LEARNING_RATE = 0.0005 # Lower learning rate for better convergence
    EPOCHS = 50 # Increased epochs
    PATIENCE = 8 # Increased patience
    
    # Paths
    base_dir = project_root
    processed_dir = os.path.join(base_dir, 'data', 'processed')
    data_path = os.path.join(processed_dir, 'X_data.npy')
    labels_path = os.path.join(processed_dir, 'y_labels.pkl')
    
    if not os.path.exists(data_path):
        print("Data not found. Please run src/preprocessing/prepare_dataset.py first.")
        return

    # 1. Prepare Data
    # Define data augmentation for training
    train_transform = Compose([
        RandomNoise(noise_level=0.05),
        RandomShift(shift_max=50)
    ])
    
    # Load dataset without transform initially
    full_dataset = ECGDataset(data_path, labels_path)
    
    # Handle small datasets gracefully
    if len(full_dataset) < 10:
        print("WARNING: Dataset is very small. Reducing batch size.")
        BATCH_SIZE = 2
        
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    
    # Split indices
    indices = list(range(len(full_dataset)))
    train_indices, val_indices = indices[:train_size], indices[train_size:]
    
    # Create subsets with appropriate transforms
    train_dataset = torch.utils.data.Subset(ECGDataset(data_path, labels_path, transform=train_transform), train_indices)
    val_dataset = torch.utils.data.Subset(ECGDataset(data_path, labels_path, transform=None), val_indices)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    print(f"Training on {len(train_dataset)} samples, Validating on {len(val_dataset)} samples.")

    # 2. Initialize Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = CNNLSTM(input_channels=12, num_classes=1).to(device)
    
    # 3. Loss and Optimizer
    criterion = nn.BCELoss()
    # AdamW adds weight decay for regularization
    # Reverted weight decay to 1e-3 to allow model to learn stronger features
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-3)
    
    # Scheduler: Reduce LR if validation loss plateaus
    # Removed verbose=True for PyTorch 2.0+ compatibility
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    
    # 4. Training Loop with Early Stopping
    best_val_loss = float('inf')
    patience_counter = 0
    
    start_time = time.time()
    
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            labels = labels.unsqueeze(1)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients in LSTM
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            running_loss += loss.item()
            predicted = (outputs > 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100 * correct / total if total > 0 else 0
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                labels = labels.unsqueeze(1)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                predicted = (outputs > 0.5).float()
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_epoch_loss = val_loss / len(val_loader) if len(val_loader) > 0 else 0
        val_acc = 100 * val_correct / val_total if val_total > 0 else 0
        
        # Step Scheduler
        scheduler.step(val_epoch_loss)
        
        print(f"Epoch [{epoch+1}/{EPOCHS}] "
              f"Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.2f}% | "
              f"Val Loss: {val_epoch_loss:.4f} Acc: {val_acc:.2f}%")
        
        # Early Stopping & Model Checkpoint
        if val_epoch_loss < best_val_loss:
            best_val_loss = val_epoch_loss
            patience_counter = 0
            # Save best model
            model_save_path = os.path.join(base_dir, 'src', 'neural_network', 'saved_model.pth')
            torch.save(model.state_dict(), model_save_path)
            print(f"--> Best model saved (Val Loss: {val_epoch_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print("Early stopping triggered.")
                break

    total_time = time.time() - start_time
    print(f"Training complete in {total_time:.0f}s. Best Val Loss: {best_val_loss:.4f}")

if __name__ == "__main__":
    train()
