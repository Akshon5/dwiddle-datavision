import os
import ast
import json
import urllib.request
import urllib.parse
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

CLASSES_20 = sorted([
    'airplane',
    'apple',
    'axe',
    'bicycle',
    'candle',
    'door',
    'key',
    'parachute',
    'book',
    'chair',
    'clock',
    'cup',
    'fish',
    'flower',
    'house',
    'scissors',
    'shoe',
    'star',
    'tree',
    'umbrella'
])

ORIGINAL_8_CLASSES = {
    'airplane', 'apple', 'axe', 'bicycle', 'candle', 'door', 'key', 'parachute'
}

LOCAL_RECON_CLASSES = {
    'cup', 'shoe'
}

LABEL_MAPPING_20 = {name: idx for idx, name in enumerate(CLASSES_20)}
REV_LABEL_MAPPING_20 = {idx: name for name, idx in LABEL_MAPPING_20.items()}

ROW_PIXELS = 784  # 28 x 28
RAW_TARGET_SAMPLES = 1600
TARGET_USABLE_PER_CLASS = 1500

ORIGINAL_DATA_DIR = 'data'
STAGE1_SAMPLE_DIR = 'dataset_recon/samples'
EXPANDED_DATA_DIR = 'data/expanded_20'
EXPANDED_RAW_DIR = 'data/expanded_20/raw'
EXPANDED_REPORTS_DIR = 'reports/expanded_20'


def fetch_and_parse_header(cat_name):
    """
    Fetches the first 256 bytes using HTTP Range and parses NumPy v1.0/v2.0 header.
    Returns: (hdr_offset, total_drawings, row_size, total_file_bytes)
    """
    url_name = urllib.parse.quote(cat_name)
    url = f'https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/{url_name}.npy'
    req = urllib.request.Request(url, headers={'Range': 'bytes=0-255'})
    with urllib.request.urlopen(req, timeout=15) as resp:
        if resp.status != 206:
            raise RuntimeError(f"Server returned HTTP {resp.status} instead of 206 for header of {cat_name}")
        content_range = resp.headers.get('Content-Range', '')
        parts = content_range.split('/')
        total_file_bytes = int(parts[-1]) if len(parts) == 2 and parts[-1].isdigit() else None
        data = resp.read()
        if data[:6] != b'\x93NUMPY':
            raise ValueError(f"Invalid NumPy magic header for {cat_name}")
        v_maj = data[6]
        if v_maj == 1:
            hdr_len = int.from_bytes(data[8:10], 'little')
            hdr_offset = 10 + hdr_len
        elif v_maj == 2:
            hdr_len = int.from_bytes(data[8:12], 'little')
            hdr_offset = 12 + hdr_len
        else:
            raise ValueError(f"Unsupported NumPy version {v_maj} for {cat_name}")
        hdr_dict = ast.literal_eval(data[10:hdr_offset].decode('latin1').strip())
        shape = hdr_dict.get('shape', ())
        if len(shape) != 2 or shape[1] != ROW_PIXELS:
            raise ValueError(f"Unexpected shape in header for {cat_name}: {shape}")
        return hdr_offset, shape[0], shape[1], total_file_bytes


def acquire_raw_class_data(cat_name):
    """
    Acquires 1,600 raw drawings for category:
    - If in ORIGINAL_8_CLASSES: reads existing data/{cat_name}_raw_1600.npy
    - If in LOCAL_RECON_CLASSES (cup, shoe): loads existing 1,000 from dataset_recon/samples/
      and downloads additional 600 via HTTP Range
    - If new: downloads 1,600 via HTTP Range
    Saves new pools in data/expanded_20/raw/
    """
    os.makedirs(EXPANDED_RAW_DIR, exist_ok=True)
    cache_path = os.path.join(EXPANDED_RAW_DIR, f"{cat_name}_raw_1600.npy")
    
    # Check if already cached in expanded raw dir
    if os.path.exists(cache_path):
        cached = np.load(cache_path)
        if cached.shape == (RAW_TARGET_SAMPLES, 28, 28) and cached.dtype == np.uint8:
            return cached, "cached_expanded_raw"
            
    # Case 1: Original 8 classes
    if cat_name in ORIGINAL_8_CLASSES:
        orig_raw_file = os.path.join(ORIGINAL_DATA_DIR, f"{cat_name}_raw_1600.npy")
        if os.path.exists(orig_raw_file):
            raw = np.load(orig_raw_file)
            if raw.shape == (RAW_TARGET_SAMPLES, 28, 28) and raw.dtype == np.uint8:
                return raw, "reused_original_stage2_pool"
                
    # Case 2: Local recon classes (cup, shoe) with 1,000 existing samples
    if cat_name in LOCAL_RECON_CLASSES:
        recon_file = os.path.join(STAGE1_SAMPLE_DIR, f"{cat_name}_1000.npy")
        if os.path.exists(recon_file):
            first_1000 = np.load(recon_file)
            if first_1000.shape == (1000, 28, 28) and first_1000.dtype == np.uint8:
                # Fetch samples 1000 to 1599 (600 drawings)
                hdr_offset, total_drawings, row_size, _ = fetch_and_parse_header(cat_name)
                start_sample = 1000
                num_to_fetch = 600
                start_byte = hdr_offset + start_sample * row_size
                expected_bytes = num_to_fetch * row_size
                end_byte = start_byte + expected_bytes - 1
                
                url_name = urllib.parse.quote(cat_name)
                url = f'https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/{url_name}.npy'
                req = urllib.request.Request(url, headers={'Range': f'bytes={start_byte}-{end_byte}'})
                with urllib.request.urlopen(req, timeout=25) as resp:
                    if resp.status != 206:
                        raise RuntimeError(f"Server returned HTTP {resp.status} for {cat_name} extension")
                    payload = resp.read()
                    if len(payload) != expected_bytes:
                        raise ValueError(f"Payload size mismatch: {len(payload)} vs {expected_bytes}")
                    ext_arr = np.frombuffer(payload, dtype=np.uint8).reshape((num_to_fetch, 28, 28))
                
                full_pool = np.concatenate([first_1000, ext_arr], axis=0)
                assert full_pool.shape == (RAW_TARGET_SAMPLES, 28, 28)
                np.save(cache_path, full_pool)
                return full_pool, "1000_local_plus_600_gcs"
                
    # Case 3: Completely new classes (download 1600 drawings from scratch)
    hdr_offset, total_drawings, row_size, _ = fetch_and_parse_header(cat_name)
    start_sample = 0
    num_to_fetch = RAW_TARGET_SAMPLES
    start_byte = hdr_offset + start_sample * row_size
    expected_bytes = num_to_fetch * row_size
    end_byte = start_byte + expected_bytes - 1
    
    url_name = urllib.parse.quote(cat_name)
    url = f'https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/{url_name}.npy'
    req = urllib.request.Request(url, headers={'Range': f'bytes={start_byte}-{end_byte}'})
    with urllib.request.urlopen(req, timeout=30) as resp:
        if resp.status != 206:
            raise RuntimeError(f"Server returned HTTP {resp.status} for {cat_name} range")
        payload = resp.read()
        if len(payload) != expected_bytes:
            raise ValueError(f"Payload size mismatch for {cat_name}: {len(payload)} vs {expected_bytes}")
        full_pool = np.frombuffer(payload, dtype=np.uint8).reshape((num_to_fetch, 28, 28))
        
    assert full_pool.shape == (RAW_TARGET_SAMPLES, 28, 28)
    np.save(cache_path, full_pool)
    return full_pool, "1600_gcs_download"


def inspect_and_filter_all():
    """
    Inspects and filters drawings across all 20 classes.
    Rejection rules:
    - nz < 10: blank or stray single dot
    - nz > 450: solid scribble block
    Returns: (filtered_dict, provenance_dict, stats_dict)
    """
    os.makedirs(EXPANDED_DATA_DIR, exist_ok=True)
    os.makedirs(EXPANDED_REPORTS_DIR, exist_ok=True)
    
    filtered_datasets = {}
    provenance = {}
    stats = {}
    rejected_samples = []
    
    print("=" * 75)
    print("ACQUIRING & FILTERING 20 OBJECT CLASSES")
    print("=" * 75)
    
    for c_name in CLASSES_20:
        raw_pool, prov = acquire_raw_class_data(c_name)
        provenance[c_name] = prov
        total_raw = len(raw_pool)
        
        nz_counts = (raw_pool > 0).sum(axis=(1, 2))
        
        count_zero = int((nz_counts == 0).sum())
        count_lt_10 = int(((nz_counts > 0) & (nz_counts < 10)).sum())
        count_gt_450 = int((nz_counts > 450).sum())
        
        keep_mask = (nz_counts >= 10) & (nz_counts <= 450)
        
        for idx in range(total_raw):
            if not keep_mask[idx] and len(rejected_samples) < 24:
                reason = "Sparse/Dot" if nz_counts[idx] < 10 else "Dense/Scribble"
                rejected_samples.append((c_name, f"{reason} (nz={nz_counts[idx]})", raw_pool[idx]))
                
        usable = raw_pool[keep_mask]
        num_usable = len(usable)
        
        if num_usable < TARGET_USABLE_PER_CLASS:
            raise RuntimeError(f"Class '{c_name}' only produced {num_usable} usable drawings, needed {TARGET_USABLE_PER_CLASS}!")
            
        final_1500 = usable[:TARGET_USABLE_PER_CLASS]
        filtered_datasets[c_name] = final_1500
        
        stats[c_name] = {
            'total_raw': total_raw,
            'provenance': prov,
            'count_zero': count_zero,
            'count_lt_10': count_lt_10,
            'count_gt_450': count_gt_450,
            'total_rejected': total_raw - num_usable,
            'usable_available': num_usable,
            'final_selected': TARGET_USABLE_PER_CLASS,
            'mean_nonzero_px': float(round(float(nz_counts.mean()), 1)),
            'std_nonzero_px': float(round(float(nz_counts.std()), 1)),
            'min_nonzero_px': int(nz_counts.min()),
            'max_nonzero_px': int(nz_counts.max())
        }
        
        print(f"[{c_name:<10}] Prov: {prov:<27} | Raw: {total_raw} | Rej: {total_raw - num_usable:2d} | Usable: {num_usable} -> Kept: {TARGET_USABLE_PER_CLASS}")
        
    return filtered_datasets, provenance, stats


def build_and_save_expanded_splits(filtered_datasets, stats, seed=42):
    """
    Partitions into stratified 70/15/15 train/val/test splits.
    Saves:
    - data/expanded_20/train.npz
    - data/expanded_20/val.npz
    - data/expanded_20/test.npz
    - data/expanded_20/split_metadata.json
    - data/expanded_20/label_mapping.json
    """
    np.random.seed(seed)
    
    n_train = 1050  # 70%
    n_val = 225    # 15%
    n_test = 225   # 15%
    assert n_train + n_val + n_test == TARGET_USABLE_PER_CLASS
    
    train_x, train_y = [], []
    val_x, val_y = [], []
    test_x, test_y = [], []
    
    split_info = {}
    
    for c_name in CLASSES_20:
        c_idx = LABEL_MAPPING_20[c_name]
        data = filtered_datasets[c_name]  # (1500, 28, 28)
        
        indices = np.arange(TARGET_USABLE_PER_CLASS)
        np.random.shuffle(indices)
        
        train_idx = indices[:n_train]
        val_idx = indices[n_train:n_train + n_val]
        test_idx = indices[n_train + n_val:]
        
        # Verify strict disjointness
        assert len(set(train_idx).intersection(set(val_idx))) == 0
        assert len(set(train_idx).intersection(set(test_idx))) == 0
        assert len(set(val_idx).intersection(set(test_idx))) == 0
        
        train_x.append(data[train_idx])
        train_y.append(np.full(n_train, c_idx, dtype=np.int64))
        
        val_x.append(data[val_idx])
        val_y.append(np.full(n_val, c_idx, dtype=np.int64))
        
        test_x.append(data[test_idx])
        test_y.append(np.full(n_test, c_idx, dtype=np.int64))
        
        split_info[c_name] = {
            'class_index': c_idx,
            'train_count': n_train,
            'val_count': n_val,
            'test_count': n_test,
            'total_count': TARGET_USABLE_PER_CLASS
        }
        
    train_x = np.concatenate(train_x, axis=0)
    train_y = np.concatenate(train_y, axis=0)
    val_x = np.concatenate(val_x, axis=0)
    val_y = np.concatenate(val_y, axis=0)
    test_x = np.concatenate(test_x, axis=0)
    test_y = np.concatenate(test_y, axis=0)
    
    # Shuffle entire split pools deterministically
    p_train = np.random.permutation(len(train_x))
    train_x, train_y = train_x[p_train], train_y[p_train]
    
    p_val = np.random.permutation(len(val_x))
    val_x, val_y = val_x[p_val], val_y[p_val]
    
    p_test = np.random.permutation(len(test_x))
    test_x, test_y = test_x[p_test], test_y[p_test]
    
    # Save NPZ arrays (saving both images/labels and x/y for universal compatibility)
    np.savez_compressed(os.path.join(EXPANDED_DATA_DIR, 'train.npz'), images=train_x, labels=train_y, x=train_x, y=train_y)
    np.savez_compressed(os.path.join(EXPANDED_DATA_DIR, 'val.npz'), images=val_x, labels=val_y, x=val_x, y=val_y)
    np.savez_compressed(os.path.join(EXPANDED_DATA_DIR, 'test.npz'), images=test_x, labels=test_y, x=test_x, y=test_y)
    
    # Save label mapping
    label_path = os.path.join(EXPANDED_DATA_DIR, 'label_mapping.json')
    with open(label_path, 'w') as f:
        json.dump(LABEL_MAPPING_20, f, indent=2)
        
    # Save split metadata
    meta = {
        'num_classes': len(CLASSES_20),
        'classes': CLASSES_20,
        'label_mapping': LABEL_MAPPING_20,
        'random_seed': seed,
        'split_ratio': {'train': 0.70, 'val': 0.15, 'test': 0.15},
        'per_class_counts': {'train': n_train, 'val': n_val, 'test': n_test, 'total': TARGET_USABLE_PER_CLASS},
        'total_counts': {'train': len(train_x), 'val': len(val_x), 'test': len(test_x), 'total': len(train_x) + len(val_x) + len(test_x)},
        'class_stats': stats,
        'split_info': split_info
    }
    
    meta_path = os.path.join(EXPANDED_DATA_DIR, 'split_metadata.json')
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)
        
    print("\n" + "=" * 75)
    print("EXPANDED 20-CLASS DATASET SAVED SUCCESSFULLY")
    print("=" * 75)
    print(f"Train split: {train_x.shape}, labels: {train_y.shape}")
    print(f"Val split:   {val_x.shape}, labels: {val_y.shape}")
    print(f"Test split:  {test_x.shape}, labels: {test_y.shape}")
    print(f"Metadata:    {meta_path}")
    print(f"Mapping:     {label_path}")
    
    return meta


def generate_visual_gallery(filtered_datasets):
    """
    Generates a 20-row x 5-column visual gallery showing 5 representative sketches
    per class from the clean filtered dataset.
    Saves to reports/expanded_20/class_samples_gallery.png
    """
    os.makedirs(EXPANDED_REPORTS_DIR, exist_ok=True)
    rows = len(CLASSES_20)
    cols = 5
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.0, rows * 1.8))
    
    for r_idx, c_name in enumerate(CLASSES_20):
        data = filtered_datasets[c_name]
        # Pick 5 diverse indices across the 1500 drawings
        sample_indices = [10, 150, 450, 800, 1200]
        for c_idx, s_idx in enumerate(sample_indices):
            ax = axes[r_idx, c_idx]
            img = data[s_idx]
            # Invert for visual viewing: white bg, black strokes
            ax.imshow(255 - img, cmap='gray', vmin=0, vmax=255)
            if c_idx == 0:
                ax.set_ylabel(f"{r_idx+1}. {c_name}", fontsize=9, fontweight='bold', rotation=0, labelpad=55, va='center')
            ax.set_xticks([])
            ax.set_yticks([])
            
    plt.suptitle("20-Class Quick, Draw! Representative Sample Gallery (5 per Class)", fontsize=14, y=0.995, fontweight='bold')
    plt.tight_layout()
    gallery_path = os.path.join(EXPANDED_REPORTS_DIR, 'class_samples_gallery.png')
    plt.savefig(gallery_path, dpi=180, bbox_inches='tight')
    plt.close()
    print(f"[GALLERY] Saved 20-class visual gallery to {gallery_path}")
    return gallery_path


if __name__ == '__main__':
    filtered, prov, stats = inspect_and_filter_all()
    meta = build_and_save_expanded_splits(filtered, stats, seed=42)
    gallery_path = generate_visual_gallery(filtered)
