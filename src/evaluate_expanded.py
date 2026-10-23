import os
import sys
import time
import json
import hashlib
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from PIL import Image

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
os.environ['MPLCONFIGDIR'] = '/tmp/matplotlib'

from src.model import QuickDrawCNN

DATA_DIR = 'data/expanded_20'
MODELS_DIR = 'models'
REPORTS_DIR = 'reports/expanded_20'


def evaluate_expanded_model():
    model_path = os.path.join(MODELS_DIR, 'expanded_cnn_20.pt')
    mapping_path = os.path.join(MODELS_DIR, 'label_mapping_20.json')
    test_data_path = os.path.join(DATA_DIR, 'test.npz')
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
    if not os.path.exists(mapping_path):
        raise FileNotFoundError(f"Label mapping not found: {mapping_path}")
    if not os.path.exists(test_data_path):
        raise FileNotFoundError(f"Test data not found: {test_data_path}")
        
    with open(mapping_path) as f:
        label_mapping = json.load(f)
    rev_mapping = {v: k for k, v in label_mapping.items()}
    class_names = [rev_mapping[i] for i in range(len(rev_mapping))]
    num_classes = len(class_names)
    assert num_classes == 20, f"Expected 20 classes, got {num_classes}"
    
    # Load model onto CPU
    checkpoint = torch.load(model_path, map_location='cpu')
    model = QuickDrawCNN(num_classes=num_classes)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Load untouched test data
    test_data = np.load(test_data_path)
    X_test_raw = test_data['images'] if 'images' in test_data else test_data['x']
    y_test = test_data['labels'] if 'labels' in test_data else test_data['y']
    
    total_test = len(y_test)
    assert total_test == 4500, f"Expected 4500 test samples, got {total_test}"
    
    # Preprocessing contract
    X_test_tensor = torch.from_numpy(X_test_raw).float().unsqueeze(1) / 255.0
    
    print("=" * 75)
    print("HELD-OUT 20-CLASS TEST SET EVALUATION")
    print("=" * 75)
    print(f"Total test drawings: {total_test} ({total_test // num_classes} per class across 20 classes)")
    print(f"Model checkpoint: {model_path} (trained for {checkpoint.get('epoch', 'N/A')} epochs)")
    
    # 1. CPU Single-Sample Inference Latency
    print("\nMeasuring single-sample CPU inference latency...")
    with torch.no_grad():
        for _ in range(50):
            _ = model(X_test_tensor[:1])
            
    latencies = []
    num_latency_trials = 1000
    with torch.no_grad():
        for i in range(num_latency_trials):
            sample = X_test_tensor[i % total_test : (i % total_test) + 1]
            t_start = time.perf_counter()
            _ = model(sample)
            t_end = time.perf_counter()
            latencies.append((t_end - t_start) * 1000.0)
            
    latencies = np.array(latencies)
    mean_lat = float(np.mean(latencies))
    median_lat = float(np.median(latencies))
    p95_lat = float(np.percentile(latencies, 95))
    std_lat = float(np.std(latencies))
    
    print(f"  CPU Latency: Mean={mean_lat:.3f} ms, Median={median_lat:.3f} ms, 95th-pct={p95_lat:.3f} ms, Std={std_lat:.3f} ms")
    
    # 2. Inference on full test set
    with torch.no_grad():
        logits = model(X_test_tensor)
        assert logits.shape == (total_test, num_classes), f"Output logits shape mismatch: {logits.shape}"
        probabilities = torch.softmax(logits, dim=1).numpy()
        predictions = np.argmax(probabilities, axis=1)
        
    correct = (predictions == y_test).sum()
    test_acc = float(correct / total_test)
    print(f"\nOverall 20-Class Test Accuracy: {test_acc * 100:.2f}% ({correct}/{total_test})")
    
    # 3. Detailed classification report
    clf_dict = classification_report(y_test, predictions, target_names=class_names, output_dict=True, digits=4)
    clf_text = classification_report(y_test, predictions, target_names=class_names, digits=4)
    print("\nClassification Report:")
    print(clf_text)
    
    macro_f1 = float(clf_dict['macro avg']['f1-score'])
    macro_prec = float(clf_dict['macro avg']['precision'])
    macro_rec = float(clf_dict['macro avg']['recall'])
    
    # 4. Confusion Matrix
    cm = confusion_matrix(y_test, predictions)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Plot normalized confusion matrix
    plt.figure(figsize=(14, 12))
    plt.imshow(cm_norm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Normalized Confusion Matrix (20 Classes)', fontsize=14, fontweight='bold', pad=15)
    plt.colorbar(fraction=0.046, pad=0.04)
    
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45, ha='right', fontsize=9)
    plt.yticks(tick_marks, class_names, fontsize=9)
    
    thresh = cm_norm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm_norm[i, j]
            if val > 0.01:
                plt.text(j, i, f"{val:.2f}",
                         ha="center", va="center",
                         color="white" if val > thresh else "black",
                         fontsize=7)
                
    plt.ylabel('True Class', fontsize=11, fontweight='bold')
    plt.xlabel('Predicted Class', fontsize=11, fontweight='bold')
    plt.tight_layout()
    cm_path = os.path.join(REPORTS_DIR, 'confusion_matrix_20.png')
    plt.savefig(cm_path, dpi=200)
    plt.close()
    print(f"\n[PLOT] Saved confusion matrix to: {cm_path}")
    
    # 5. Top misclassified examples
    misclassified_indices = np.where(predictions != y_test)[0]
    print(f"Total misclassified test drawings: {len(misclassified_indices)} out of {total_test} ({len(misclassified_indices)/total_test*100:.2f}%)")
    
    if len(misclassified_indices) > 0:
        # Sort by confidence of wrong prediction
        wrong_confidences = [probabilities[idx, predictions[idx]] for idx in misclassified_indices]
        sorted_wrong_order = np.argsort(wrong_confidences)[::-1]
        top_wrong_indices = misclassified_indices[sorted_wrong_order[:24]]
        
        cols = 6
        rows = 4
        fig, axes = plt.subplots(rows, cols, figsize=(15, 10))
        axes = axes.flatten()
        
        for plot_idx, sample_idx in enumerate(top_wrong_indices):
            ax = axes[plot_idx]
            true_label = class_names[y_test[sample_idx]]
            pred_label = class_names[predictions[sample_idx]]
            conf = probabilities[sample_idx, predictions[sample_idx]] * 100
            
            # Invert for white bg
            ax.imshow(255 - X_test_raw[sample_idx], cmap='gray', vmin=0, vmax=255)
            ax.set_title(f"True: {true_label}\nPred: {pred_label} ({conf:.0f}%)", fontsize=8, color='darkred', fontweight='bold')
            ax.axis('off')
            
        plt.suptitle("Top Confident Misclassifications on 20-Class Held-Out Test Set", fontsize=13, fontweight='bold')
        plt.tight_layout()
        misclass_path = os.path.join(REPORTS_DIR, 'misclassified_examples_20.png')
        plt.savefig(misclass_path, dpi=200)
        plt.close()
        print(f"[PLOT] Saved misclassified gallery to: {misclass_path}")
        
    # 6. Save results JSON
    results = {
        'model_checkpoint': model_path,
        'label_mapping_path': mapping_path,
        'num_classes': num_classes,
        'classes': class_names,
        'test_samples_total': total_test,
        'test_samples_per_class': total_test // num_classes,
        'test_accuracy': float(test_acc),
        'macro_precision': float(macro_prec),
        'macro_recall': float(macro_rec),
        'macro_f1': float(macro_f1),
        'cpu_latency_ms': {
            'mean': mean_lat,
            'median': median_lat,
            'p95': p95_lat,
            'std': std_lat
        },
        'per_class_metrics': {
            c: {
                'precision': float(clf_dict[c]['precision']),
                'recall': float(clf_dict[c]['recall']),
                'f1_score': float(clf_dict[c]['f1-score']),
                'support': int(clf_dict[c]['support'])
            } for c in class_names
        },
        'confusion_matrix': cm.tolist()
    }
    
    results_path = os.path.join(REPORTS_DIR, 'evaluation_results_20.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n[OUTPUT] Saved evaluation metrics to: {results_path}")
    
    # 7. Check original 8-class files integrity via MD5
    original_hashes = {
        'data/train.npz': '99def32bf871f4b178e03b0a7aaf2770',
        'data/val.npz': '9807d10c5d29a212aa90e8f1be41b5a2',
        'data/test.npz': '09b1b3be13b6f71e85f1900b9bbf6d2a',
        'data/split_metadata.json': '0829ef282848cd63045920f825e07ed9',
        'models/baseline_cnn.pt': 'b5e8888d7261a7c1889d9d1bdb5b5d12',
        'models/label_mapping.json': 'b3556e85f9bbd10c92af147aec407036'
    }
    print("\nVerifying original 8-class files integrity...")
    for fpath, expected_h in original_hashes.items():
        with open(fpath, 'rb') as fp:
            actual_h = hashlib.md5(fp.read()).hexdigest()
        assert actual_h == expected_h, f"ORIGINAL FILE COMPROMISED: {fpath}"
    print("  -> ALL ORIGINAL 8-CLASS ASSETS ARE 100% UNCHANGED!")
    
    return results


if __name__ == '__main__':
    evaluate_expanded_model()
