# STAGE B: EXPANDED 20-CLASS DATASET PREPARATION REPORT

**Project:** Draw to Survive: An AI-Powered Drawing Adventure  
**Date:** October 14, 2026  
**Scope:** Controlled Expansion to 20 Object Classes (Dataset Preparation Only)  
**Status:** **STAGE B COMPLETE & VALIDATED**

---

## 1. Executive Summary

In Stage B, a balanced 20-class dataset was prepared, filtered, and partitioned into stratified splits with zero modifications to the existing 8-class project files.

- **Total Classes:** Exactly **20 unique categories** (8 original + 12 new, with `shoe` selected instead of `hammer`).
- **Usable Target:** Exactly **1,500 usable drawings per class** (30,000 drawings total).
- **Stratified Partition:** Exactly 70% train (21,000 drawings), 15% val (4,500 drawings), 15% test (4,500 drawings).
- **Data Integrity:** 0 blank or corrupt drawings, 0 sample leakage across splits, and 100% preservation of original 8-class files (verified via MD5 hashes).
- **Representative Visual Gallery:** Saved to [`reports/expanded_20/class_samples_gallery.png`](file:///Users/akshon/Desktop/datavision/reports/expanded_20/class_samples_gallery.png).

---

## 2. Category Inventory, Provenance & Filtering Statistics

All 20 classes drew from a raw pool of 1,600 drawings to ensure a comfortable buffer above the 1,500 usable requirement. Filtering removed drawings with fewer than 10 stroke pixels (empty / stray clicks) or more than 450 stroke pixels (solid scribble blocks).

| # | Class Name | Provenance Source | Raw Pool | Filtered / Rej | Usable Available | Kept for Dataset | Mean Stroke Pixels |
| :-: | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **01** | `airplane` | Reused Stage 2 local raw pool | 1,600 | 0 | 1,600 | **1,500** | 204.7 px |
| **02** | `apple` | Reused Stage 2 local raw pool | 1,600 | 1 | 1,599 | **1,500** | 202.9 px |
| **03** | `axe` | Reused Stage 2 local raw pool | 1,600 | 1 | 1,599 | **1,500** | 178.2 px |
| **04** | `bicycle` | Reused Stage 2 local raw pool | 1,600 | 0 | 1,600 | **1,500** | 220.8 px |
| **05** | `book` | Remote GCS Range download (1,600) | 1,600 | 8 | 1,592 | **1,500** | 282.2 px |
| **06** | `candle` | Reused Stage 2 local raw pool | 1,600 | 1 | 1,599 | **1,500** | 152.1 px |
| **07** | `chair` | Remote GCS Range download (1,600) | 1,600 | 0 | 1,600 | **1,500** | 178.2 px |
| **08** | `clock` | Remote GCS Range download (1,600) | 1,600 | 1 | 1,599 | **1,500** | 247.7 px |
| **09** | `cup` | 1,000 local recon + 600 GCS Range | 1,600 | 0 | 1,600 | **1,500** | 233.6 px |
| **10** | `door` | Reused Stage 2 local raw pool | 1,600 | 0 | 1,600 | **1,500** | 202.5 px |
| **11** | `fish` | Remote GCS Range download (1,600) | 1,600 | 0 | 1,600 | **1,500** | 196.1 px |
| **12** | `flower` | Remote GCS Range download (1,600) | 1,600 | 1 | 1,599 | **1,500** | 219.0 px |
| **13** | `house` | Remote GCS Range download (1,600) | 1,600 | 0 | 1,600 | **1,500** | 238.2 px |
| **14** | `key` | Reused Stage 2 local raw pool | 1,600 | 0 | 1,600 | **1,500** | 179.7 px |
| **15** | `parachute` | Reused Stage 2 local raw pool | 1,600 | 1 | 1,599 | **1,500** | 243.1 px |
| **16** | `scissors` | Remote GCS Range download (1,600) | 1,600 | 0 | 1,600 | **1,500** | 231.9 px |
| **17** | `shoe` | 1,000 local recon + 600 GCS Range | 1,600 | 2 | 1,598 | **1,500** | 190.3 px |
| **18** | `star` | Remote GCS Range download (1,600) | 1,600 | 0 | 1,600 | **1,500** | 224.8 px |
| **19** | `tree` | Remote GCS Range download (1,600) | 1,600 | 0 | 1,600 | **1,500** | 228.1 px |
| **20** | `umbrella` | Remote GCS Range download (1,600) | 1,600 | 0 | 1,600 | **1,500** | 202.3 px |
| **TOTAL** | **20 Classes** | **8 local + 2 augmented + 10 GCS** | **32,000** | **16** | **31,984** | **30,000** | **217.8 px** |

---

## 3. Stratified Split Partitioning (Seed 42)

The 30,000 filtered drawings were partitioned using a fixed random seed (`seed = 42`) into strictly disjoint splits:

* **Training Set:** **21,000 drawings** (1,050 per class, 70.0%) $\rightarrow$ `data/expanded_20/train.npz` (5.2 MB)
* **Validation Set:** **4,500 drawings** (225 per class, 15.0%) $\rightarrow$ `data/expanded_20/val.npz` (1.1 MB)
* **Held-Out Test Set:** **4,500 drawings** (225 per class, 15.0%) $\rightarrow$ `data/expanded_20/test.npz` (1.1 MB)
* **Total Storage:** ~7.4 MB compressed NPZ data.

### Label Mapping (`data/expanded_20/label_mapping.json`)
Alphabetically mapped from index 0 to 19:
```json
{
  "airplane": 0, "apple": 1, "axe": 2, "bicycle": 3, "book": 4,
  "candle": 5, "chair": 6, "clock": 7, "cup": 8, "door": 9,
  "fish": 10, "flower": 11, "house": 12, "key": 13, "parachute": 14,
  "scissors": 15, "shoe": 16, "star": 17, "tree": 18, "umbrella": 19
}
```

---

## 4. Rigorous Quality & Integrity Verification

An automated verification protocol tested all 9 requirements:

1. **Unique Classes:** Exactly 20 distinct classes verified against `label_mapping.json`.
2. **Array Shapes:** `train_x` = $(21000, 28, 28)$, `val_x` = $(4500, 28, 28)$, `test_x` = $(4500, 28, 28)$.
3. **Data Types & Pixel Bounds:** `uint8` arrays with pixel intensity bounded strictly in $[0, 255]$.
4. **Label Types & Bounds:** `int64` targets strictly bounded in $[0, 19]$.
5. **No Invalid Drawings:** Zero drawings with stroke pixels $< 10$ or $> 450$ in any split.
6. **Per-Class Balance:** Exactly 1,050 train, 225 val, 225 test for all 20 classes.
7. **Disjointness & Zero Leakage:** SHA-256 hash sets verified 0 overlap between train-val, train-test, and val-test.
8. **Original 8-Class Files Preservation:** MD5 checksums of all original assets are identical:
   - `data/train.npz`: `99def32bf871f4b178e03b0a7aaf2770` (MATCH)
   - `data/val.npz`: `9807d10c5d29a212aa90e8f1be41b5a2` (MATCH)
   - `data/test.npz`: `09b1b3be13b6f71e85f1900b9bbf6d2a` (MATCH)
   - `data/split_metadata.json`: `0829ef282848cd63045920f825e07ed9` (MATCH)
   - `models/baseline_cnn.pt`: `b5e8888d7261a7c1889d9d1bdb5b5d12` (MATCH)
   - `models/label_mapping.json`: `b3556e85f9bbd10c92af147aec407036` (MATCH)
9. **Visual Inspection Gallery:** Saved at [`reports/expanded_20/class_samples_gallery.png`](file:///Users/akshon/Desktop/datavision/reports/expanded_20/class_samples_gallery.png).
