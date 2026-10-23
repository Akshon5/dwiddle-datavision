import os
import sys
import time
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
os.environ['MPLCONFIGDIR'] = '/tmp/matplotlib'

from src.model import QuickDrawCNN

SEED = 42
BATCH_SIZE = 64
EPOCHS = 15
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4

DATA_DIR = 'data/expanded_20'
MODELS_DIR = 'models'
REPORTS_DIR = 'reports/expanded_20'


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
    # Support both images and x
    images = data['images'] if 'images' in data else data['x']
    labels = data['labels'] if 'labels' in data else data['y']
    
    x_tensor = torch.from_numpy(images).float().unsqueeze(1) / 255.0
    y_tensor = torch.from_numpy(labels).long()
    
    return TensorDataset(x_tensor, y_tensor)


def train_model():
    set_seed(SEED)
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)
    
    # Save a copy of the 20-class label mapping into models/label_mapping_20.json
    src_mapping = os.path.join(DATA_DIR, 'label_mapping.json')
    dst_mapping = os.path.join(MODELS_DIR, 'label_mapping_20.json')
    with open(src_mapping, 'r') as f:
        mapping = json.load(f)
    with open(dst_mapping, 'w') as f:
        json.dump(mapping, f, indent=2)
    print(f"Saved 20-class label mapping to: {dst_mapping}")
    
    num_classes = len(mapping)
    assert num_classes == 20, f"Expected 20 classes, got {num_classes}"
    
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device for training: {device}")
    
    train_dataset = load_dataset_split(os.path.join(DATA_DIR, 'train.npz'))
    val_dataset = load_dataset_split(os.path.join(DATA_DIR, 'val.npz'))
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    print(f"Train samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")
    
    model = QuickDrawCNN(num_classes=num_classes).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,}")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'learning_rates': [],
        'epoch_times': []
    }
    
    best_val_acc = 0.0
    best_epoch = 0
    best_model_path = os.path.join(MODELS_DIR, 'expanded_cnn_20.pt')
    
    start_total_time = time.time()
    
    print("\n" + "=" * 75)
    print("STARTING 20-CLASS CNN TRAINING (15 EPOCHS)")
    print("=" * 75)
    
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
        
        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step(epoch_val_loss)
        
        epoch_time = time.time() - t0
        
        history['train_loss'].append(epoch_train_loss)
        history['train_acc'].append(epoch_train_acc)
        history['val_loss'].append(epoch_val_loss)
        history['val_acc'].append(epoch_val_acc)
        history['learning_rates'].append(current_lr)
        history['epoch_times'].append(epoch_time)
        
        is_best = epoch_val_acc > best_val_acc
        if is_best:
            best_val_acc = epoch_val_acc
            best_epoch = epoch
            # Save best checkpoint
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': best_val_acc,
                'val_loss': epoch_val_loss,
                'num_classes': num_classes,
                'classes': [mapping[k] for k in sorted(mapping, key=mapping.get)],
                'hyperparameters': {
                    'seed': SEED,
                    'batch_size': BATCH_SIZE,
                    'learning_rate': LEARNING_RATE,
                    'weight_decay': WEIGHT_DECAY,
                    'epochs': EPOCHS,
                    'device': str(device)
                }
            }, best_model_path)
            star = " ★ [BEST SAVED]"
        else:
            star = ""
            
        print(f"Epoch [{epoch:2d}/{EPOCHS:2d}] ({epoch_time:4.1f}s) | "
              f"Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc * 100:5.2f}% | "
              f"Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc * 100:5.2f}% | "
              f"LR: {current_lr:.1e}{star}")
              
    total_time = time.time() - start_total_time
    print("=" * 75)
    print(f"Training completed in {total_time:.1f}s ({total_time / 60:.2f} mins).")
    print(f"Best Validation Accuracy: {best_val_acc * 100:.2f}% at Epoch {best_epoch}.")
    print(f"Best checkpoint saved to: {best_model_path}")
    
    # Save training history JSON
    history_path = os.path.join(REPORTS_DIR, 'training_history_20.json')
    history['total_training_time_seconds'] = total_time
    history['best_epoch'] = best_epoch
    history['best_val_acc'] = best_val_acc
    history['device'] = str(device)
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"Saved training history to: {history_path}")
    
    # Plot training curves
    plt.figure(figsize=(12, 5))
    
    # Loss plot
    plt.subplot(1, 2, 1)
    plt.plot(range(1, EPOCHS + 1), history['train_loss'], label='Train Loss', color='blue', marker='o')
    plt.plot(range(1, EPOCHS + 1), history['val_loss'], label='Val Loss', color='orange', marker='s')
    plt.axvline(best_epoch, color='green', linestyle='--', label=f'Best Epoch ({best_epoch})')
    plt.title('20-Class QuickDrawCNN: Loss Curves', fontsize=12, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Cross-Entropy Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Accuracy plot
    plt.subplot(1, 2, 2)
    plt.plot(range(1, EPOCHS + 1), [a * 100 for a in history['train_acc']], label='Train Accuracy', color='blue', marker='o')
    plt.plot(range(1, EPOCHS + 1), [a * 100 for a in history['val_acc']], label='Val Accuracy', color='orange', marker='s')
    plt.axvline(best_epoch, color='green', linestyle='--', label=f'Best Epoch ({best_epoch})')
    plt.title('20-Class QuickDrawCNN: Accuracy Curves', fontsize=12, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    curves_path = os.path.join(REPORTS_DIR, 'training_curves_20.png')
    plt.savefig(curves_path, dpi=200)
    plt.close()
    print(f"Saved training curves to: {curves_path}")
    
    return history


if __name__ == '__main__':
    train_model()
