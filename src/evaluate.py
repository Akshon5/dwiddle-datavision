import os
import sys

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
os.environ['MPLCONFIGDIR'] = '/tmp/matplotlib'

import time
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from PIL import Image

from src.model import QuickDrawCNN

DATA_DIR = 'data'
MODELS_DIR = 'models'
REPORTS_DIR = 'reports'


def evaluate_test_set():
    model_path = os.path.join(MODELS_DIR, 'baseline_cnn.pt')
    mapping_path = os.path.join(MODELS_DIR, 'label_mapping.json')
    test_data_path = os.path.join(DATA_DIR, 'test.npz')
    
    with open(mapping_path) as f:
        label_mapping = json.load(f)
    rev_mapping = {v: k for k, v in label_mapping.items()}
    class_names = [rev_mapping[i] for i in range(len(rev_mapping))]
    
    # Load model
    checkpoint = torch.load(model_path, map_location='cpu')
    model = QuickDrawCNN(num_classes=len(class_names))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Load test data
    test_data = np.load(test_data_path)
    X_test_raw = test_data['images']  # (1800, 28, 28), uint8
    y_test = test_data['labels']      # (1800,), int64
    
    # Preprocessing contract
    X_test_tensor = torch.from_numpy(X_test_raw).float().unsqueeze(1) / 255.0
    
    print("=" * 65)
    print("HELD-OUT TEST SET EVALUATION")
    print("=" * 65)
    print(f"Total test drawings: {len(y_test)} ({len(y_test)//len(class_names)} per class)")
    
    # 1. CPU Inference Latency Measurement
    print("\nMeasuring CPU single-sample inference latency...")
    # Warmup
    with torch.no_grad():
        for _ in range(50):
            _ = model(X_test_tensor[:1])
            
    latencies = []
    num_latency_trials = 1000
    with torch.no_grad():
        for i in range(num_latency_trials):
            sample = X_test_tensor[i % len(X_test_tensor): (i % len(X_test_tensor)) + 1]
            t_start = time.perf_counter()
            _ = model(sample)
            t_end = time.perf_counter()
            latencies.append((t_end - t_start) * 1000.0)  # ms
            
    latencies = np.array(latencies)
    mean_lat = float(np.mean(latencies))
    median_lat = float(np.median(latencies))
    p95_lat = float(np.percentile(latencies, 95))
    std_lat = float(np.std(latencies))
    
    print(f"  CPU Latency per sketch: Mean={mean_lat:.3f} ms, Median={median_lat:.3f} ms, 95th-pct={p95_lat:.3f} ms, Std={std_lat:.3f} ms")
    
    # 2. Run inference on full test set
    with torch.no_grad():
        logits = model(X_test_tensor)
        probabilities = torch.softmax(logits, dim=1).numpy()
        predictions = np.argmax(probabilities, axis=1)
        
    # Accuracy
    correct = (predictions == y_test).sum()
    total = len(y_test)
    test_acc = float(correct / total)
    print(f"\nOverall Test Accuracy: {test_acc * 100:.2f}% ({correct}/{total})")
    
    # Classification Report
    clf_dict = classification_report(y_test, predictions, target_names=class_names, output_dict=True, digits=4)
    clf_text = classification_report(y_test, predictions, target_names=class_names, digits=4)
    print("\nClassification Report:")
    print(clf_text)
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, predictions)
    print("Confusion Matrix:")
    print(cm)
    
    # Plot Confusion Matrix
    plot_confusion_matrix(cm, class_names, test_acc)
    
    # 3. Misclassification and Error Analysis
    misclassified_indices = np.where(predictions != y_test)[0]
    num_errors = len(misclassified_indices)
    print(f"\nTotal misclassifications: {num_errors} out of {total} ({num_errors/total*100:.2f}%)")
    
    error_pairs = {}
    for idx in misclassified_indices:
        true_c = class_names[y_test[idx]]
        pred_c = class_names[predictions[idx]]
        pair = f"{true_c} -> {pred_c}"
        error_pairs[pair] = error_pairs.get(pair, 0) + 1
        
    sorted_errors = sorted(error_pairs.items(), key=lambda x: x[1], reverse=True)
    print("\nTop Confusion Patterns (True Class -> Predicted Class):")
    for pair, cnt in sorted_errors[:8]:
        print(f"  {pair:25}: {cnt} instances")
        
    # Save visual gallery of misclassified sketches
    plot_misclassified_gallery(X_test_raw, y_test, predictions, probabilities, misclassified_indices, class_names)
    
    # Model footprint
    model_size_bytes = os.path.getsize(model_path)
    total_params = sum(p.numel() for p in model.parameters())
    
    results = {
        'test_accuracy': test_acc,
        'macro_precision': clf_dict['macro avg']['precision'],
        'macro_recall': clf_dict['macro avg']['recall'],
        'macro_f1': clf_dict['macro avg']['f1-score'],
        'weighted_f1': clf_dict['weighted avg']['f1-score'],
        'per_class_metrics': {
            c: {
                'precision': clf_dict[c]['precision'],
                'recall': clf_dict[c]['recall'],
                'f1_score': clf_dict[c]['f1-score'],
                'support': int(clf_dict[c]['support'])
            } for c in class_names
        },
        'confusion_matrix': cm.tolist(),
        'top_confusion_pairs': sorted_errors,
        'cpu_latency_ms': {
            'mean': mean_lat,
            'median': median_lat,
            'p95': p95_lat,
            'std': std_lat,
            'trials': num_latency_trials,
            'measurement_method': 'Single-sample CPU inference with time.perf_counter() over 1000 trials after warmup'
        },
        'model_footprint': {
            'checkpoint_path': model_path,
            'file_size_kb': round(model_size_bytes / 1024, 2),
            'file_size_mb': round(model_size_bytes / (1024 * 1024), 2),
            'total_parameters': total_params
        }
    }
    
    with open(os.path.join(REPORTS_DIR, 'evaluation_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
        
    print(f"\nSaved evaluation metrics to {os.path.join(REPORTS_DIR, 'evaluation_results.json')}")
    return results


def plot_confusion_matrix(cm, class_names, test_acc):
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=class_names, yticklabels=class_names,
           title=f'Test Confusion Matrix (Accuracy: {test_acc*100:.2f}%)',
           ylabel='True Class',
           xlabel='Predicted Class')
    
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontweight='bold')
            
    fig.tight_layout()
    cm_path = os.path.join(REPORTS_DIR, 'confusion_matrix.png')
    plt.savefig(cm_path, dpi=200)
    plt.close()
    print(f"[PLOT] Saved confusion matrix to {cm_path}")


def plot_misclassified_gallery(X_raw, y_true, y_pred, probs, error_indices, class_names):
    if len(error_indices) == 0:
        return
    
    # Pick up to 16 diverse errors
    sample_errors = error_indices[:16]
    n_samples = len(sample_errors)
    cols = 4
    rows = (n_samples + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(10, rows * 2.6))
    axes = axes.flatten()
    
    for i in range(len(axes)):
        if i < n_samples:
            idx = sample_errors[i]
            true_lbl = class_names[y_true[idx]]
            pred_lbl = class_names[y_pred[idx]]
            conf = probs[idx][y_pred[idx]] * 100
            
            # QuickDraw original is black bg (0) and white stroke (255)
            # Display inverted (white bg, black stroke) for visual readability
            img = 255 - X_raw[idx]
            axes[i].imshow(img, cmap='gray')
            axes[i].set_title(f"True: {true_lbl}\nPred: {pred_lbl} ({conf:.0f}%)", fontsize=9, color='darkred')
            axes[i].axis('off')
        else:
            axes[i].axis('off')
            
    plt.tight_layout()
    gallery_path = os.path.join(REPORTS_DIR, 'misclassified_examples.png')
    plt.savefig(gallery_path, dpi=200)
    plt.close()
    print(f"[PLOT] Saved misclassified examples gallery to {gallery_path}")


if __name__ == '__main__':
    evaluate_test_set()
