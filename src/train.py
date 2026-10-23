import os
import sys

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
os.environ['MPLCONFIGDIR'] = '/tmp/matplotlib'

import time
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt

from src.model import QuickDrawCNN

SEED = 42
BATCH_SIZE = 64
EPOCHS = 15
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4

DATA_DIR = 'data'
MODELS_DIR = 'models'
REPORTS_DIR = 'reports'


def set_seed(seed=SEED):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_dataset_split(npz_path):
    """
    Loads data and applies the preprocessing contract:
    - uint8 [0, 255] -> float32 [0.0, 1.0]
    - (N, 28, 28) -> (N, 1, 28, 28)
    - Background is 0.0 (dark), strokes are >0.0 (light)
    """
    data = np.load(npz_path)
    images = data['images']  # (N, 28, 28), uint8
    labels = data['labels']  # (N,), int64
    
    # Preprocessing contract
    x_tensor = torch.from_numpy(images).float().unsqueeze(1) / 255.0
    y_tensor = torch.from_numpy(labels).long()
    
    return TensorDataset(x_tensor, y_tensor)


def train_model():
    set_seed(SEED)
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)
    
    # Device selection: Apple Silicon MPS if available, else CPU
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device for training: {device}")
    
    train_dataset = load_dataset_split(os.path.join(DATA_DIR, 'train.npz'))
    val_dataset = load_dataset_split(os.path.join(DATA_DIR, 'val.npz'))
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    print(f"Train samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")
    
    model = QuickDrawCNN(num_classes=8).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'epoch_times': []
    }
    
    best_val_acc = 0.0
    best_model_path = os.path.join(MODELS_DIR, 'baseline_cnn.pt')
    
    start_total_time = time.time()
    
    print("\n" + "=" * 65)
    print("STARTING BASELINE CNN TRAINING")
    print("=" * 65)
    
    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        
        # Training phase
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct_train += (preds == labels).sum().item()
            total_train += labels.size(0)
            
        epoch_train_loss = running_loss / total_train
        epoch_train_acc = correct_train / total_train
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                correct_val += (preds == labels).sum().item()
                total_val += labels.size(0)
                
        epoch_val_loss = val_loss / total_val
        epoch_val_acc = correct_val / total_val
        epoch_time = time.time() - t0
        
        scheduler.step(epoch_val_loss)
        
        history['train_loss'].append(epoch_train_loss)
        history['train_acc'].append(epoch_train_acc)
        history['val_loss'].append(epoch_val_loss)
        history['val_acc'].append(epoch_val_acc)
        history['epoch_times'].append(epoch_time)
        
        improved = ""
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': epoch_val_acc,
                'val_loss': epoch_val_loss,
                'num_classes': 8
            }, best_model_path)
            improved = " [BEST MODEL SAVED]"
            
        print(f"Epoch [{epoch:02d}/{EPOCHS:02d}] ({epoch_time:.2f}s) | "
              f"Train Loss: {epoch_train_loss:.4f}, Acc: {epoch_train_acc*100:6.2f}% | "
              f"Val Loss: {epoch_val_loss:.4f}, Acc: {epoch_val_acc*100:6.2f}%{improved}")
        
    total_duration = time.time() - start_total_time
    print("=" * 65)
    print(f"Training completed in {total_duration:.2f} seconds ({total_duration/60:.2f} min)")
    print(f"Best Validation Accuracy: {best_val_acc*100:.2f}% saved to {best_model_path}")
    
    # Save training history JSON
    history['total_training_duration_sec'] = total_duration
    history['best_val_acc'] = best_val_acc
    with open(os.path.join(REPORTS_DIR, 'training_history.json'), 'w') as f:
        json.dump(history, f, indent=2)
        
    # Plot Training & Validation Curves
    plot_curves(history)
    return best_model_path, history


def plot_curves(history):
    epochs = range(1, len(history['train_loss']) + 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    
    # Loss plot
    ax1.plot(epochs, history['train_loss'], 'b-o', label='Training Loss', linewidth=2)
    ax1.plot(epochs, history['val_loss'], 'r-s', label='Validation Loss', linewidth=2)
    ax1.set_title('Training & Validation Loss', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Epoch', fontsize=10)
    ax1.set_ylabel('Cross-Entropy Loss', fontsize=10)
    ax1.legend(loc='upper right')
    ax1.grid(True, linestyle='--', alpha=0.6)
    
    # Accuracy plot
    ax2.plot(epochs, [a * 100 for a in history['train_acc']], 'b-o', label='Training Accuracy', linewidth=2)
    ax2.plot(epochs, [a * 100 for a in history['val_acc']], 'r-s', label='Validation Accuracy', linewidth=2)
    ax2.set_title('Training & Validation Accuracy', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Epoch', fontsize=10)
    ax2.set_ylabel('Accuracy (%)', fontsize=10)
    ax2.legend(loc='lower right')
    ax2.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    curve_path = os.path.join(REPORTS_DIR, 'training_curves.png')
    plt.savefig(curve_path, dpi=200)
    plt.close()
    print(f"[PLOT] Saved training curves to {curve_path}")


if __name__ == '__main__':
    train_model()
