import os
import ast
import json
import csv
import urllib.request
import urllib.parse
import numpy as np
from PIL import Image

CATEGORIES = [
    'key', 'hammer', 'axe', 'ladder', 'bridge', 'flashlight', 'door',
    'airplane', 'parachute', 'screwdriver', 'candle', 'lantern',
    'shoe', 'apple', 'cup', 'bicycle'
]

SAMPLES_PER_CAT = 1000
ROW_PIXELS = 784  # 28 x 28
SAMPLE_DIR = 'dataset_recon/samples'
GALLERY_DIR = 'dataset_recon/gallery'

os.makedirs(SAMPLE_DIR, exist_ok=True)
os.makedirs(GALLERY_DIR, exist_ok=True)


def fetch_and_parse_header(cat_name):
    """
    Fetches the first 256 bytes of the .npy file using HTTP Range request.
    Verifies HTTP 206 status, Content-Range, magic bytes, and parses header dictionary.
    Returns: (hdr_offset, total_drawings, row_size, total_file_bytes)
    """
    url_name = urllib.parse.quote(cat_name)
    url = f'https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/{url_name}.npy'
    
    req = urllib.request.Request(url, headers={'Range': 'bytes=0-255'})
    with urllib.request.urlopen(req, timeout=15) as resp:
        # 1. Strictly verify HTTP 206 Partial Content
        if resp.status != 206:
            raise RuntimeError(f"Server did not honor Range request (HTTP {resp.status}) for {cat_name}")
        
        # 2. Verify Content-Range header
        content_range = resp.headers.get('Content-Range', '')
        if not content_range or not content_range.startswith('bytes'):
            raise RuntimeError(f"Invalid or missing Content-Range header: {content_range}")
        
        parts = content_range.split('/')
        total_file_bytes = int(parts[-1]) if len(parts) == 2 and parts[-1].isdigit() else None
        
        data = resp.read()
        if len(data) < 16:
            raise ValueError(f"Received truncated header data ({len(data)} bytes) for {cat_name}")
        
        # 3. Verify NumPy magic
        if data[:6] != b'\x93NUMPY':
            raise ValueError(f"Invalid NumPy magic header in file for {cat_name}: {data[:6]}")
        
        v_maj = data[6]
        v_min = data[7]
        
        if v_maj == 1:
            hdr_len = int.from_bytes(data[8:10], 'little')
            total_hdr_offset = 10 + hdr_len
        elif v_maj == 2:
            hdr_len = int.from_bytes(data[8:12], 'little')
            total_hdr_offset = 12 + hdr_len
        else:
            raise ValueError(f"Unsupported NumPy format version {v_maj}.{v_min} for {cat_name}")
        
        hdr_raw = data[10:total_hdr_offset]
        hdr_dict = ast.literal_eval(hdr_raw.decode('latin1').strip())
        
        shape = hdr_dict.get('shape', ())
        descr = hdr_dict.get('descr', '')
        fortran_order = hdr_dict.get('fortran_order', False)
        
        if len(shape) != 2 or shape[1] != ROW_PIXELS:
            raise ValueError(f"Unexpected shape in header for {cat_name}: {shape}")
        if descr != '|u1':
            raise ValueError(f"Unexpected dtype descriptor for {cat_name}: {descr} (expected '|u1' uint8)")
        if fortran_order:
            raise ValueError(f"Unexpected Fortran order in {cat_name}")
        
        total_drawings = shape[0]
        row_size = shape[1]
        
        return total_hdr_offset, total_drawings, row_size, total_file_bytes


def acquire_category_sample(cat_name, num_samples=SAMPLES_PER_CAT):
    """
    Acquires sample drawings, reusing existing valid local files if available.
    Strictly verifies HTTP status 206, byte counts, and array dimensions.
    """
    sample_file = os.path.join(SAMPLE_DIR, f"{cat_name}_{num_samples}.npy")
    
    # First, always fetch and verify header metadata from server
    hdr_offset, total_drawings, row_size, total_file_bytes = fetch_and_parse_header(cat_name)
    
    # Check if a valid local file already exists to avoid redundant download
    if os.path.exists(sample_file):
        try:
            arr = np.load(sample_file)
            if arr.shape == (num_samples, 28, 28) and arr.dtype == np.uint8:
                print(f"  [CACHE HIT] Loaded existing verified sample for '{cat_name}': shape={arr.shape}")
                return arr, total_drawings, total_file_bytes, hdr_offset, os.path.getsize(sample_file)
            else:
                print(f"  Local file for '{cat_name}' has mismatched shape/dtype ({arr.shape}, {arr.dtype}). Re-downloading...")
        except Exception as e:
            print(f"  Local file for '{cat_name}' could not be loaded ({e}). Re-downloading...")
    
    # Download missing sample via HTTP Range
    url_name = urllib.parse.quote(cat_name)
    url = f'https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/{url_name}.npy'
    
    start_byte = hdr_offset
    expected_bytes = num_samples * row_size
    end_byte = hdr_offset + expected_bytes - 1
    
    req = urllib.request.Request(url, headers={'Range': f'bytes={start_byte}-{end_byte}'})
    with urllib.request.urlopen(req, timeout=20) as resp:
        # Strictly verify HTTP 206 Partial Content
        if resp.status != 206:
            raise RuntimeError(f"Server returned HTTP {resp.status} instead of 206 for range {start_byte}-{end_byte}")
        
        cr = resp.headers.get('Content-Range', '')
        if not cr or f'bytes {start_byte}-{end_byte}' not in cr:
            raise RuntimeError(f"Unexpected Content-Range returned: {cr}")
        
        payload = resp.read()
        if len(payload) != expected_bytes:
            raise ValueError(f"Payload length mismatch: expected {expected_bytes} bytes, got {len(payload)} bytes")
        
        # Deserialization and shape validation
        flat_arr = np.frombuffer(payload, dtype=np.uint8)
        if flat_arr.size != num_samples * 28 * 28:
            raise ValueError(f"Array size mismatch: {flat_arr.size}")
        
        arr = flat_arr.reshape((num_samples, 28, 28))
        np.save(sample_file, arr)
        saved_bytes = os.path.getsize(sample_file)
        print(f"  [DOWNLOADED] Successfully fetched '{cat_name}': {expected_bytes} bytes -> {sample_file}")
        
        return arr, total_drawings, total_file_bytes, hdr_offset, saved_bytes


def main():
    print("=" * 70)
    print("STAGE 1: VERIFIED QUICK DRAW DATASET RECONNAISSANCE")
    print(f"Target: {len(CATEGORIES)} candidate categories, {SAMPLES_PER_CAT} samples each")
    print("=" * 70)
    
    successful_categories = {}
    inventory_records = []
    
    recommendation_map = {
        'apple': ('Recommended', 'Circular body with distinct stem/leaf; food/health game mechanic; visually unique.'),
        'airplane': ('Recommended', 'Distinct cruciform/swept-wing profile; flight/travel escape mechanic; easily recognized.'),
        'bicycle': ('Recommended', 'Two separated wheels with frame and handlebars; iconic topology; land vehicle mechanic.'),
        'candle': ('Recommended', 'Vertical cylinder with teardrop flame; light/dungeon dark-room mechanic; distinct from tools.'),
        'door': ('Recommended', 'Rectangular frame with doorknob; classic barrier/escape mechanic; clean geometric edges.'),
        'key': ('Recommended', 'Round head with long stem and notched teeth; essential unlock mechanic; high visual contrast.'),
        'parachute': ('Recommended', 'Hemispheric canopy with converging suspension cords; aerial descent challenge; unmistakable silhouette.'),
        'axe': ('Recommended', 'Wedge blade with handle; obstacle-clearing/tool mechanic; pick one edged tool (avoiding hammer/screwdriver confusion).'),
        'hammer': ('Avoid / Confusion Risk', 'High visual confusion with axe and screwdriver (handle + head); players draw basic stick-T shape.'),
        'screwdriver': ('Avoid / Confusion Risk', 'Confusable with key, candle, hammer, and wand; long thin shaft with small handle.'),
        'flashlight': ('Avoid / Alternative to Candle', 'Confusable with candle, hammer, and microphone; cylindrical body with flared end.'),
        'lantern': ('Avoid / Redundant', 'Complex caged shape; frequently drawn as candle, lamp, or bottle; excessive stroke variance.'),
        'ladder': ('Avoid / High Confusion', 'Parallel vertical lines with rungs; frequently confused with door, bridge, and railway tracks.'),
        'bridge': ('Avoid / High Variance', 'Enormous visual diversity (arch, suspension, beam); often confused with ladder or bench.'),
        'cup': ('Backup / Good Alternative', 'Mug with handle; clean shape, but sometimes resembles cylinder/pot or bucket.'),
        'shoe': ('Backup / Alternative', 'L-shaped side profile, but high variance across sneakers, boots, and high heels.')
    }
    
    for cat in CATEGORIES:
        print(f"\nProcessing category: '{cat}'...")
        try:
            arr, total_drawings, total_file_bytes, hdr_offset, local_bytes = acquire_category_sample(cat, SAMPLES_PER_CAT)
            successful_categories[cat] = arr
            
            # Pixel & stroke analytics
            nz_counts = (arr > 0).sum(axis=(1, 2))
            blank_drawings = int((nz_counts == 0).sum())
            near_blank_drawings = int(((nz_counts > 0) & (nz_counts < 15)).sum())
            dense_drawings = int((nz_counts > 450).sum())
            
            rec_status, rec_rationale = recommendation_map.get(cat, ('Evaluated', ''))
            
            inventory_records.append({
                'category': cat,
                'status': 'Verified & Sampled',
                'recommendation': rec_status,
                'header_offset_bytes': hdr_offset,
                'total_dataset_examples': total_drawings,
                'full_file_mb': round(total_file_bytes / (1024 * 1024), 2) if total_file_bytes else None,
                'sample_count': SAMPLES_PER_CAT,
                'sample_file_kb': round(local_bytes / 1024, 1),
                'image_shape': '28x28',
                'dtype': str(arr.dtype),
                'pixel_min': int(arr.min()),
                'pixel_max': int(arr.max()),
                'pixel_mean': float(round(arr.mean(), 2)),
                'blank_drawings': blank_drawings,
                'near_blank_drawings': near_blank_drawings,
                'overly_dense_drawings': dense_drawings,
                'avg_stroke_pixels': round(float(nz_counts.mean()), 1),
                'rationale': rec_rationale
            })
            print(f"  ✓ Verified: {total_drawings:,} drawings in full file, sample avg stroke pixels: {nz_counts.mean():.1f}, blanks: {blank_drawings}")
        except Exception as e:
            print(f"  ✗ Failed category '{cat}': {e}")
            rec_status, rec_rationale = recommendation_map.get(cat, ('Error', ''))
            inventory_records.append({
                'category': cat,
                'status': f'Error: {e}',
                'recommendation': 'Unavailable',
                'header_offset_bytes': None,
                'total_dataset_examples': None,
                'full_file_mb': None,
                'sample_count': 0,
                'sample_file_kb': 0,
                'image_shape': None,
                'dtype': None,
                'pixel_min': None,
                'pixel_max': None,
                'pixel_mean': None,
                'blank_drawings': None,
                'near_blank_drawings': None,
                'overly_dense_drawings': None,
                'avg_stroke_pixels': None,
                'rationale': f'Error encountered during inspection: {e}'
            })
            
    # Write Category Inventory CSV
    csv_path = 'dataset_recon/category_inventory.csv'
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(inventory_records[0].keys()))
        writer.writeheader()
        writer.writerows(inventory_records)
    print(f"\n[OUTPUT] Category inventory saved to {csv_path}")
    
    # Generate visual galleries ONLY for successfully verified categories
    print("\nGenerating visual inspection galleries...")
    
    def render_grid(cat_list, cols=10, scale=3, pad=2, invert=True):
        rows = len(cat_list)
        cell_size = 28 * scale
        grid_w = cols * cell_size + (cols + 1) * pad
        label_w = 110
        total_w = grid_w + label_w
        total_h = rows * cell_size + (rows + 1) * pad
        
        bg_col = 255 if invert else 0
        grid_img = Image.new('L', (total_w, total_h), color=bg_col)
        
        for r_idx, c_name in enumerate(cat_list):
            if c_name not in successful_categories:
                continue
            arr = successful_categories[c_name]
            y = pad + r_idx * (cell_size + pad)
            
            for col_idx in range(cols):
                if col_idx >= len(arr):
                    break
                cell_arr = arr[col_idx]
                if invert:
                    cell_arr = 255 - cell_arr
                pil_cell = Image.fromarray(cell_arr, mode='L').resize((cell_size, cell_size), resample=Image.NEAREST)
                x = label_w + pad + col_idx * (cell_size + pad)
                grid_img.paste(pil_cell, (x, y))
                
        return grid_img

    # 1. Recommended 8 categories gallery (if all recommended exist)
    recommended_8 = ['apple', 'airplane', 'bicycle', 'candle', 'door', 'key', 'parachute', 'axe']
    verified_rec = [c for c in recommended_8 if c in successful_categories]
    
    if verified_rec:
        rec_inv = render_grid(verified_rec, cols=10, scale=3, invert=True)
        rec_inv_path = os.path.join(GALLERY_DIR, 'gallery_recommended_8_inverted.png')
        rec_inv.save(rec_inv_path)
        print(f"  [SAVED] Recommended 8 gallery (inverted): {rec_inv_path}")
        
        rec_orig = render_grid(verified_rec, cols=10, scale=3, invert=False)
        rec_orig_path = os.path.join(GALLERY_DIR, 'gallery_recommended_8_original.png')
        rec_orig.save(rec_orig_path)
        print(f"  [SAVED] Recommended 8 gallery (original): {rec_orig_path}")
        
    # 2. All verified candidates gallery
    verified_all = [c for c in CATEGORIES if c in successful_categories]
    if verified_all:
        all_grid = render_grid(verified_all, cols=8, scale=2, invert=True)
        all_grid_path = os.path.join(GALLERY_DIR, 'gallery_all_candidates.png')
        all_grid.save(all_grid_path)
        print(f"  [SAVED] All verified candidates gallery: {all_grid_path}")
        
    # 3. Quality & Anomaly Inspection sheet (min stroke vs median vs dense)
    quality_samples = []
    for c_name in verified_rec:
        arr = successful_categories[c_name]
        nz = (arr > 0).sum(axis=(1, 2))
        min_i = int(np.argmin(nz))
        max_i = int(np.argmax(nz))
        med_i = int(np.argsort(nz)[len(nz)//2])
        quality_samples.extend([
            (c_name, f"Sparse ({nz[min_i]}px)", arr[min_i]),
            (c_name, f"Typical ({nz[med_i]}px)", arr[med_i]),
            (c_name, f"Dense ({nz[max_i]}px)", arr[max_i])
        ])
        
    if quality_samples:
        q_cols = 6
        q_rows = (len(quality_samples) + q_cols - 1) // q_cols
        q_scale = 3
        q_cell = 28 * q_scale
        q_pad = 4
        q_w = q_cols * (q_cell + q_pad) + q_pad
        q_h = q_rows * (q_cell + q_pad) + q_pad
        q_sheet = Image.new('L', (q_w, q_h), color=255)
        
        for idx, (c_name, label, drawing) in enumerate(quality_samples):
            col = idx % q_cols
            row = idx // q_cols
            cell_img = Image.fromarray(255 - drawing, mode='L').resize((q_cell, q_cell), resample=Image.NEAREST)
            q_sheet.paste(cell_img, (q_pad + col * (q_cell + q_pad), q_pad + row * (q_cell + q_pad)))
            
        q_path = os.path.join(GALLERY_DIR, 'gallery_quality_inspection.png')
        q_sheet.save(q_path)
        print(f"  [SAVED] Quality inspection gallery: {q_path}")
        
    print("\nReconnaissance process complete!")

if __name__ == '__main__':
    main()
