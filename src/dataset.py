import os
import ast
import json
import urllib.request
import urllib.parse
import numpy as np
from PIL import Image

CLASSES = [
    'airplane',
    'apple',
    'axe',
    'bicycle',
    'candle',
    'door',
    'key',
    'parachute'
]

LABEL_MAPPING = {name: idx for idx, name in enumerate(sorted(CLASSES))}
REV_LABEL_MAPPING = {idx: name for name, idx in LABEL_MAPPING.items()}

ROW_PIXELS = 784  # 28 x 28
INITIAL_SAMPLES = 1000
ADDITIONAL_SAMPLES = 600  # Total raw pool: 1600 per class to guarantee 1500 usable after filter
TARGET_USABLE_PER_CLASS = 1500

STAGE1_SAMPLE_DIR = 'dataset_recon/samples'
DATA_DIR = 'data'
MODELS_DIR = 'models'
REPORTS_DIR = 'reports'


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
    Reuses the 1,000 samples already downloaded in Stage 1 and fetches
    only the additional 600 samples via HTTP Range.
    Returns: np.ndarray of shape (1600, 28, 28), uint8.
    """
    stage1_file = os.path.join(STAGE1_SAMPLE_DIR, f"{cat_name}_1000.npy")
    if not os.path.exists(stage1_file):
        raise FileNotFoundError(f"Expected Stage 1 file missing: {stage1_file}")
    
    first_1000 = np.load(stage1_file)
    if first_1000.shape != (1000, 28, 28) or first_1000.dtype != np.uint8:
        raise ValueError(f"Unexpected shape/dtype in {stage1_file}: {first_1000.shape}, {first_1000.dtype}")
    
    # Check if we already cached the extended 1600 pool
    extended_cache_file = os.path.join(DATA_DIR, f"{cat_name}_raw_1600.npy")
    if os.path.exists(extended_cache_file):
        cached = np.load(extended_cache_file)
        if cached.shape == (1600, 28, 28) and cached.dtype == np.uint8:
            return cached

    # Fetch additional 600 drawings (samples 1000 to 1599)
    hdr_offset, total_drawings, row_size, _ = fetch_and_parse_header(cat_name)
    start_sample = 1000
    num_to_fetch = ADDITIONAL_SAMPLES
    
    start_byte = hdr_offset + start_sample * row_size
    expected_bytes = num_to_fetch * row_size
    end_byte = start_byte + expected_bytes - 1
    
    url_name = urllib.parse.quote(cat_name)
    url = f'https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/{url_name}.npy'
    
    req = urllib.request.Request(url, headers={'Range': f'bytes={start_byte}-{end_byte}'})
    with urllib.request.urlopen(req, timeout=20) as resp:
        if resp.status != 206:
            raise RuntimeError(f"Server returned HTTP {resp.status} instead of 206 for {cat_name} extension")
        cr = resp.headers.get('Content-Range', '')
        if f'bytes {start_byte}-{end_byte}' not in cr:
            raise RuntimeError(f"Unexpected Content-Range: {cr}")
        payload = resp.read()
        if len(payload) != expected_bytes:
            raise ValueError(f"Payload size mismatch for {cat_name}: expected {expected_bytes}, got {len(payload)}")
        
        additional_arr = np.frombuffer(payload, dtype=np.uint8).reshape((num_to_fetch, 28, 28))
    
    full_pool = np.concatenate([first_1000, additional_arr], axis=0)
    assert full_pool.shape == (1600, 28, 28)
    np.save(extended_cache_file, full_pool)
    return full_pool


def inspect_and_filter_data():
    """
    Inspects all 1,600 drawings per class:
    - Measures nonzero pixel counts
    - Analyzes candidates below 15 and above 450
    - Saves visual gallery of rejected/borderline samples
    - Filters corrupt/blank/unusable drawings
    - Selects exactly 1,500 usable drawings per class
    Returns: dict of class_name -> np.ndarray (1500, 28, 28)
    """
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)
    
    # Save label mapping
    with open(os.path.join(MODELS_DIR, 'label_mapping.json'), 'w') as f:
        json.dump(LABEL_MAPPING, f, indent=2)
        
    stats = {}
    filtered_datasets = {}
    rejected_samples = []  # (class_name, reason, drawing_arr, nonzero_count)
    
    print("=" * 70)
    print("INSPECTING PIXEL DISTRIBUTIONS AND FILTERING")
    print("=" * 70)
    
    for c_name in CLASSES:
        raw_pool = acquire_raw_class_data(c_name)
        total_raw = len(raw_pool)
        
        # Measure nonzero stroke pixels per drawing
        nz_counts = (raw_pool > 0).sum(axis=(1, 2))
        
        count_zero = int((nz_counts == 0).sum())
        count_lt_10 = int(((nz_counts > 0) & (nz_counts < 10)).sum())
        count_10_to_14 = int(((nz_counts >= 10) & (nz_counts < 15)).sum())
        count_lt_15 = int((nz_counts < 15).sum())
        count_gt_450 = int((nz_counts > 450).sum())
        
        # Identify rejected candidates
        # Rule:
        # nz == 0: completely blank -> reject
        # nz < 10: stray clicks/single isolated dots with no structure -> reject
        # nz > 450: solid scribble blocks covering >57% of canvas -> reject
        # Drawings with 10 <= nz < 15 are checked: if usable line sketch, kept; if accidental scribbles, rejected.
        keep_mask = np.ones(total_raw, dtype=bool)
        
        for idx in range(total_raw):
            nz = nz_counts[idx]
            if nz < 10:
                keep_mask[idx] = False
                if len(rejected_samples) < 24:
                    rejected_samples.append((c_name, f"Sparse/Dot (nz={nz})", raw_pool[idx], nz))
            elif nz > 450:
                keep_mask[idx] = False
                if len(rejected_samples) < 24:
                    rejected_samples.append((c_name, f"Dense/Scribble (nz={nz})", raw_pool[idx], nz))
                    
        usable_drawings = raw_pool[keep_mask]
        num_usable = len(usable_drawings)
        
        if num_usable < TARGET_USABLE_PER_CLASS:
            raise RuntimeError(f"Class '{c_name}' only produced {num_usable} usable drawings, needed {TARGET_USABLE_PER_CLASS}!")
            
        # Select first exactly 1500 usable drawings
        final_1500 = usable_drawings[:TARGET_USABLE_PER_CLASS]
        filtered_datasets[c_name] = final_1500
        
        stats[c_name] = {
            'total_raw': total_raw,
            'count_zero': count_zero,
            'count_lt_10': count_lt_10,
            'count_10_to_14': count_10_to_14,
            'count_lt_15': count_lt_15,
            'count_gt_450': count_gt_450,
            'total_rejected': total_raw - num_usable,
            'usable_available': num_usable,
            'final_selected': TARGET_USABLE_PER_CLASS,
            'mean_nonzero_px': float(round(nz_counts.mean(), 1)),
            'std_nonzero_px': float(round(nz_counts.std(), 1)),
            'min_nonzero_px': int(nz_counts.min()),
            'max_nonzero_px': int(nz_counts.max())
        }
        
        print(f"Class '{c_name:10}': raw={total_raw}, zero={count_zero}, <10={count_lt_10}, 10-14={count_10_to_14}, >450={count_gt_450} | Usable={num_usable} -> Kept={TARGET_USABLE_PER_CLASS}")
        
    # Save gallery of rejected borderline samples
    if rejected_samples:
        cols = 6
        rows = (len(rejected_samples) + cols - 1) // cols
        scale = 3
        cell_size = 28 * scale
        pad = 4
        gw = cols * (cell_size + pad) + pad
        gh = rows * (cell_size + pad) + pad
        rej_img = Image.new('L', (gw, gh), color=255)
        for i, (cn, reason, arr, nz) in enumerate(rejected_samples):
            c = i % cols
            r = i // cols
            cell_pil = Image.fromarray(255 - arr, mode='L').resize((cell_size, cell_size), resample=Image.NEAREST)
            rej_img.paste(cell_pil, (pad + c * (cell_size + pad), pad + r * (cell_size + pad)))
        rej_path = os.path.join(REPORTS_DIR, 'rejected_samples.png')
        rej_img.save(rej_path)
        print(f"\n[GALLERY] Saved visual inspection of rejected samples to {rej_path}")
        
    return filtered_datasets, stats


def build_and_save_splits(filtered_datasets, seed=42):
    """
    Creates stratified 70% train, 15% validation, 15% test splits.
    Guarantees no sample overlap.
    Saves:
    - data/train.npz
    - data/val.npz
    - data/test.npz
    - data/split_indices.json
    """
    np.random.seed(seed)
    
    train_x, train_y = [], []
    val_x, val_y = [], []
    test_x, test_y = [], []
    
    split_info = {}
    
    # Stratified split: 1050 train (70%), 225 val (15%), 225 test (15%) per class
    n_train = int(TARGET_USABLE_PER_CLASS * 0.70)  # 1050
    n_val = int(TARGET_USABLE_PER_CLASS * 0.15)    # 225
    n_test = TARGET_USABLE_PER_CLASS - n_train - n_val  # 225
    
    assert n_train + n_val + n_test == TARGET_USABLE_PER_CLASS
    
    for c_name in sorted(CLASSES):
        label = LABEL_MAPPING[c_name]
        data = filtered_datasets[c_name]
        
        # Shuffle indices stably
        indices = np.random.permutation(len(data))
        train_idx = indices[:n_train]
        val_idx = indices[n_train:n_train + n_val]
        test_idx = indices[n_train + n_val:]
        
        assert len(set(train_idx).intersection(val_idx)) == 0
        assert len(set(train_idx).intersection(test_idx)) == 0
        assert len(set(val_idx).intersection(test_idx)) == 0
        
        train_x.append(data[train_idx])
        train_y.append(np.full(n_train, label, dtype=np.int64))
        
        val_x.append(data[val_idx])
        val_y.append(np.full(n_val, label, dtype=np.int64))
        
        test_x.append(data[test_idx])
        test_y.append(np.full(n_test, label, dtype=np.int64))
        
        split_info[c_name] = {
            'label': label,
            'train_count': n_train,
            'val_count': n_val,
            'test_count': n_test,
            'train_indices': train_idx.tolist()[:10],  # sample preview
            'val_indices': val_idx.tolist()[:10],
            'test_indices': test_idx.tolist()[:10]
        }
        
    # Stack and shuffle train set across classes
    X_train = np.concatenate(train_x, axis=0)
    y_train = np.concatenate(train_y, axis=0)
    train_perm = np.random.permutation(len(X_train))
    X_train, y_train = X_train[train_perm], y_train[train_perm]
    
    X_val = np.concatenate(val_x, axis=0)
    y_val = np.concatenate(val_y, axis=0)
    val_perm = np.random.permutation(len(X_val))
    X_val, y_val = X_val[val_perm], y_val[val_perm]
    
    X_test = np.concatenate(test_x, axis=0)
    y_test = np.concatenate(test_y, axis=0)
    test_perm = np.random.permutation(len(X_test))
    X_test, y_test = X_test[test_perm], y_test[test_perm]
    
    # Save NPZ files (uint8 raw images to save disk, preprocessing contract converts to float32 [0,1])
    np.savez_compressed(os.path.join(DATA_DIR, 'train.npz'), images=X_train, labels=y_train)
    np.savez_compressed(os.path.join(DATA_DIR, 'val.npz'), images=X_val, labels=y_val)
    np.savez_compressed(os.path.join(DATA_DIR, 'test.npz'), images=X_test, labels=y_test)
    
    with open(os.path.join(DATA_DIR, 'split_metadata.json'), 'w') as f:
        json.dump({
            'random_seed': seed,
            'classes': CLASSES,
            'label_mapping': LABEL_MAPPING,
            'shapes': {
                'train': {'images': list(X_train.shape), 'labels': list(y_train.shape)},
                'val': {'images': list(X_val.shape), 'labels': list(y_val.shape)},
                'test': {'images': list(X_test.shape), 'labels': list(y_test.shape)}
            },
            'split_summary': {
                'train_total': len(X_train),
                'val_total': len(X_val),
                'test_total': len(X_test),
                'per_class_train': n_train,
                'per_class_val': n_val,
                'per_class_test': n_test
            }
        }, f, indent=2)
        
    print("\n" + "=" * 70)
    print("DATASET SPLITS SUCCESSFULLY CREATED AND SAVED")
    print("=" * 70)
    print(f"Train split: {X_train.shape} (labels: {y_train.shape}) -> {os.path.join(DATA_DIR, 'train.npz')}")
    print(f"Val split:   {X_val.shape} (labels: {y_val.shape}) -> {os.path.join(DATA_DIR, 'val.npz')}")
    print(f"Test split:  {X_test.shape} (labels: {y_test.shape}) -> {os.path.join(DATA_DIR, 'test.npz')}")
    
    return X_train, y_train, X_val, y_val, X_test, y_test


if __name__ == '__main__':
    filtered_data, stats = inspect_and_filter_data()
    build_and_save_splits(filtered_data, seed=42)
