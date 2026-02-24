"""
UNSW-NB15 Dataset Loader (Memory-Efficient Lazy Windowing)
Reference: https://research.unsw.edu.au/projects/unsw-nb15-dataset

Follows the same interface as CICIDS2017 loader:
- Lazy windowing (windows generated on-the-fly)
- Strict temporal split (70/10/20)
- Leakage-free scaling (scaler fitted on train only)
- Binary + multi-class label support
"""

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler, LabelEncoder
import os
import joblib
import gc


class UNSWNB15Dataset(Dataset):
    """
    Memory-efficient UNSW-NB15 loader with lazy windowing.
    
    Expected files in root_dir:
        UNSW-NB15_1.csv, UNSW-NB15_2.csv, UNSW-NB15_3.csv, UNSW-NB15_4.csv
        OR
        UNSW_NB15_training-set.csv, UNSW_NB15_testing-set.csv
    """
    
    ATTACK_CATEGORIES = [
        'Normal', 'Fuzzers', 'Analysis', 'Backdoor', 'DoS',
        'Exploits', 'Generic', 'Reconnaissance', 'Shellcode', 'Worms'
    ]
    
    def __init__(self, root_dir="data/raw/UNSW-NB15", split="train", binary=True, seq_len=50):
        self.root_dir = root_dir
        self.split = split
        self.binary = binary
        self.seq_len = seq_len
        self.scaler_path = "outputs/scaler_unswnb15.pkl"
        
        self.X_raw, self.y_raw, self.attack_labels = self._load_process_split()
        
        if len(self.X_raw) >= self.seq_len:
            self.num_windows = len(self.X_raw) - self.seq_len + 1
        else:
            self.num_windows = 0
    
    def _load_process_split(self):
        if not os.path.exists(self.root_dir):
            os.makedirs(self.root_dir, exist_ok=True)
        
        # Try different file naming conventions
        # IMPORTANT: Prioritize training/testing CSVs — they have proper headers
        # (label, attack_cat columns). The raw _1 to _4 CSVs lack headers and
        # will cause all samples to be mislabeled as Attack.
        possible_files = [
            # Training/testing split (has headers with 'label' and 'attack_cat')
            ["UNSW_NB15_training-set.csv", "UNSW_NB15_testing-set.csv"],
            # Standard 4-part split (raw, no headers — needs column mapping)
            [f"UNSW-NB15_{i}.csv" for i in range(1, 5)],
            # Alternative names
            ["UNSW-NB15_features.csv"],
        ]
        
        available_files = []
        for file_set in possible_files:
            found = [f for f in file_set if os.path.exists(os.path.join(self.root_dir, f))]
            if found:
                available_files = found
                break
        
        if not available_files:
            print(f"⚠ No UNSW-NB15 files found in {self.root_dir}.")
            print(f"  Expected files like: UNSW-NB15_1.csv or UNSW_NB15_training-set.csv")
            return np.array([]), np.array([]), np.array([])
        
        print(f"[{self.split.upper()}] Loading UNSW-NB15...")
        print(f"Files: {available_files}")
        
        dfs = []
        for f in available_files:
            print(f"  -> Processing {f}...")
            try:
                chunk_iter = pd.read_csv(
                    os.path.join(self.root_dir, f),
                    encoding='latin1',
                    chunksize=100000,
                    low_memory=False
                )
                for chunk in chunk_iter:
                    chunk.columns = [c.strip() for c in chunk.columns]
                    dfs.append(chunk)
                gc.collect()
            except Exception as e:
                print(f"Error loading {f}: {e}")
        
        if not dfs:
            return np.array([]), np.array([]), np.array([])
        
        full_df = pd.concat(dfs, ignore_index=True)
        del dfs
        gc.collect()
        
        print(f"Loaded {len(full_df)} total rows.")
        
        # Label processing
        # UNSW-NB15 has 'attack_cat' (category) and 'label' (binary) columns
        attack_cat_col = None
        label_col = None
        
        for c in full_df.columns:
            cl = c.strip().lower()
            if cl == 'attack_cat':
                attack_cat_col = c
            elif cl == 'label':
                label_col = c
        
        # Extract attack category labels for per-attack analysis
        if attack_cat_col and attack_cat_col in full_df.columns:
            attack_labels = full_df[attack_cat_col].fillna('Normal').astype(str).str.strip().values
        else:
            attack_labels = np.array(['Unknown'] * len(full_df))
        
        # Binary labels
        if label_col and label_col in full_df.columns:
            if self.binary:
                y_all = full_df[label_col].values.astype(np.int64)
            else:
                le = LabelEncoder()
                y_all = le.fit_transform(attack_labels)
        else:
            # Fallback: derive from attack_cat
            y_all = np.where(np.char.lower(attack_labels.astype(str)) == 'normal', 0, 1)
        
        print(f"Class Distribution: Benign={np.sum(y_all == 0)}, Attack={np.sum(y_all == 1)}")
        
        # Drop non-feature columns
        drop_cols = {'id', 'label', 'attack_cat', 'attack_category'}
        cols_to_drop = [c for c in full_df.columns if c.strip().lower() in drop_cols]
        full_df.drop(columns=cols_to_drop, inplace=True, errors='ignore')
        
        # Convert categorical columns to numeric
        for col in full_df.select_dtypes(include=['object']).columns:
            try:
                full_df[col] = pd.to_numeric(full_df[col], errors='coerce')
            except:
                le = LabelEncoder()
                full_df[col] = le.fit_transform(full_df[col].astype(str))
        
        # Convert to float32
        X_all = full_df.values.astype(np.float32)
        
        # Clean NaN/Inf
        for i in range(X_all.shape[1]):
            col = X_all[:, i]
            col[np.isnan(col)] = 0.0
            col[np.isinf(col)] = 0.0
        
        del full_df
        gc.collect()
        
        # Strict temporal split
        total_len = len(X_all)
        train_end = int(total_len * 0.70)
        val_end = int(total_len * 0.80)
        
        print(f"Total samples: {total_len}")
        print(f"Train: 0-{train_end}, Val: {train_end}-{val_end}, Test: {val_end}-{total_len}")
        
        if self.split == 'train':
            X_part = X_all[:train_end]
            y_part = y_all[:train_end]
            atk_part = attack_labels[:train_end]
        elif self.split == 'val':
            X_part = X_all[train_end:val_end]
            y_part = y_all[train_end:val_end]
            atk_part = attack_labels[train_end:val_end]
        elif self.split == 'test':
            X_part = X_all[val_end:]
            y_part = y_all[val_end:]
            atk_part = attack_labels[val_end:]
        else:
            raise ValueError(f"Unknown split: {self.split}")
        
        del X_all, y_all, attack_labels
        gc.collect()
        
        # Leakage-free scaling
        if self.split == 'train':
            scaler = StandardScaler()
            X_part = scaler.fit_transform(X_part)
            os.makedirs("outputs", exist_ok=True)
            joblib.dump(scaler, self.scaler_path)
            print(f"Saved scaler to {self.scaler_path}")
        else:
            if os.path.exists(self.scaler_path):
                scaler = joblib.load(self.scaler_path)
                X_part = scaler.transform(X_part)
            else:
                raise FileNotFoundError("Scaler not found. Run training split first.")
        
        print(f"[{self.split.upper()}] Raw shape: {X_part.shape}")
        print(f"[OK] Will generate {max(0, len(X_part) - self.seq_len + 1)} windows lazily")
        
        return X_part, y_part, atk_part
    
    def __len__(self):
        return self.num_windows
    
    def __getitem__(self, idx):
        if idx >= self.num_windows:
            raise IndexError(f"Index {idx} out of range for {self.num_windows} windows")
        
        window = self.X_raw[idx:idx + self.seq_len]
        label = self.y_raw[idx + self.seq_len - 1]
        
        return torch.FloatTensor(window), torch.tensor(label, dtype=torch.float32)
    
    def get_attack_types(self):
        """Return unique attack types in this split."""
        if len(self.attack_labels) == 0:
            return []
        return list(set(self.attack_labels))
    
    def get_attack_label(self, idx):
        """Get the attack category label for a specific window."""
        if idx >= self.num_windows:
            raise IndexError
        return self.attack_labels[idx + self.seq_len - 1]
    
    def get_feature_count(self):
        """Return number of features."""
        if len(self.X_raw) == 0:
            return 0
        return self.X_raw.shape[1]
