#!/usr/bin/env python
"""
Comprehensive Benchmark Runner
===============================
Single entry point for all experiments.

Usage:
    # Run full benchmark (all datasets, all models, 5 seeds)
    python run_benchmark.py

    # Run specific datasets/models
    python run_benchmark.py --datasets CICIDS2017 --models Mamba LSTM --seeds 42 123

    # Quick test (1 seed, 2 epochs)
    python run_benchmark.py --datasets CICIDS2017 --models Mamba LSTM --seeds 42 --quick
"""

import argparse
import json
import os
import sys
import time
import traceback

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from datasets.dataset_factory import get_dataset, list_datasets
from models.benchmark_models import get_model, count_params, list_models
from evaluate import evaluate_model, measure_efficiency, per_attack_metrics, compute_confidence_interval
from utils.config_loader import load_config
from utils.reproducibility import set_seeds


# Per-model learning rate multipliers — sensitive architectures need lower LR
MODEL_LR_SCALE = {
    'Mamba': 0.3,        # SSM is sensitive to large updates
    'Transformer': 0.3,  # Self-attention gradients can explode
    'CNN-LSTM': 0.5,     # Hybrid arch needs moderate LR
    'TCN': 0.5,          # Dilated convs can be unstable
    'LSTM': 0.5,         # Reduced from 1.0 — collapsed on seeds 123/456
    'GRU': 0.5,          # Reduced from 1.0 — collapsed on seeds 123/456
}

GRAD_CLIP_NORM = {
    'Mamba': 0.5,
    'Transformer': 0.5,
    'CNN-LSTM': 0.5,
    'TCN': 1.0,
    'LSTM': 1.0,
    'GRU': 0.5,          # GRU was unstable, tighter clipping
}

# All recurrent/sequential models get warmup to prevent majority-class collapse
WARMUP_MODELS = {'Mamba', 'Transformer', 'LSTM', 'GRU'}
WARMUP_EPOCHS = 5


def train_model(model, train_loader, val_loader, config, device, pos_weight=None, model_name='Unknown'):
    """
    Train a model with early stopping, class-weighted loss, NaN protection,
    per-model learning rate, and linear warmup.
    
    Args:
        pos_weight: weight for positive (attack) class. If None, equal weighting.
        model_name: name of the model (for per-model LR/clipping settings).
    
    Returns:
        dict with best_val_f1, training_history, train_time_sec
    """
    epochs = config['training']['epochs']
    base_lr = config['training']['learning_rate']
    patience = config['training']['early_stopping']['patience']
    
    # Apply per-model LR scaling
    lr_scale = MODEL_LR_SCALE.get(model_name, 1.0)
    lr = base_lr * lr_scale
    clip_norm = GRAD_CLIP_NORM.get(model_name, 1.0)
    print(f"  LR: {lr:.6f} (base={base_lr} × {lr_scale} for {model_name}), grad_clip={clip_norm}")
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)
    
    # Class-weighted loss to handle imbalanced datasets
    if pos_weight is not None:
        pw = torch.tensor([pos_weight], dtype=torch.float32).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pw)
        print(f"  Using class-weighted loss (pos_weight={pos_weight:.2f})")
    else:
        criterion = nn.BCEWithLogitsLoss()
    
    best_val_f1 = 0.0
    best_state = None
    patience_counter = 0
    history = []
    
    t_start = time.time()
    
    for epoch in range(epochs):
        # === Linear warmup (only for sensitive models) ===
        use_warmup = model_name in WARMUP_MODELS
        if use_warmup and epoch < WARMUP_EPOCHS:
            warmup_factor = (epoch + 1) / WARMUP_EPOCHS
            for pg in optimizer.param_groups:
                pg['lr'] = lr * warmup_factor
        
        # === Training ===
        model.train()
        train_loss = 0
        n_batches = 0
        nan_batches = 0
        
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x).squeeze()
            loss = criterion(logits, y)
            
            # NaN protection: skip bad batches
            if torch.isnan(loss) or torch.isinf(loss):
                nan_batches += 1
                if nan_batches > 10:
                    print(f"  ⚠ Too many NaN batches ({nan_batches}), reducing LR by 10x")
                    for pg in optimizer.param_groups:
                        pg['lr'] *= 0.1
                    nan_batches = 0  # Reset counter after LR reduction
                continue
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_norm)
            optimizer.step()
            train_loss += loss.item()
            n_batches += 1
        
        if nan_batches > 0:
            print(f"  ⚠ {nan_batches} NaN batches skipped in epoch {epoch+1}")
        avg_train_loss = train_loss / max(n_batches, 1)
        
        # === Validation ===
        model.eval()
        val_preds = []
        val_targets = []
        val_loss = 0
        n_val = 0
        
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x).squeeze()
                loss = criterion(logits, y)
                preds = (torch.sigmoid(logits) > 0.5).float()
                val_preds.extend(preds.cpu().numpy())
                val_targets.extend(y.cpu().numpy())
                val_loss += loss.item()
                n_val += 1
        
        from sklearn.metrics import f1_score, accuracy_score
        val_f1 = f1_score(val_targets, val_preds, zero_division=0)
        val_acc = accuracy_score(val_targets, val_preds)
        avg_val_loss = val_loss / max(n_val, 1)
        
        # Track best model
        is_best = val_f1 > best_val_f1
        if is_best:
            best_val_f1 = val_f1
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
        
        scheduler.step(val_f1)
        
        history.append({
            'epoch': epoch + 1,
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
            'val_acc': float(val_acc),
            'val_f1': float(val_f1),
            'is_best': is_best,
        })
        
        # Print progress
        marker = " *" if is_best else ""
        print(f"  Epoch {epoch+1:3d}/{epochs} | "
              f"TrLoss: {avg_train_loss:.4f} | "
              f"VlLoss: {avg_val_loss:.4f} | "
              f"VlAcc: {val_acc:.4f} | "
              f"VlF1: {val_f1:.4f}{marker}")
        
        # Early stopping
        if patience_counter >= patience:
            print(f"  Early stopping at epoch {epoch+1} (patience={patience})")
            break
    
    train_time = time.time() - t_start
    
    # Load best model state
    if best_state is not None:
        model.load_state_dict(best_state)
    
    return {
        'best_val_f1': float(best_val_f1),
        'history': history,
        'train_time_sec': float(train_time),
        'epochs_trained': len(history),
    }


def run_single_experiment(dataset_name, model_name, seed, config, device):
    """
    Run a single experiment: train + evaluate one model on one dataset with one seed.
    
    Returns:
        dict with all results
    """
    print(f"\n{'='*70}")
    print(f"  Dataset: {dataset_name} | Model: {model_name} | Seed: {seed}")
    print(f"{'='*70}")
    
    # Set seed
    set_seeds(seed)
    
    seq_len = config['dataset']['seq_len']
    batch_size = config['training']['batch_size']
    d_model = config['model']['d_model']
    n_layers = config['model']['n_layers']
    
    # Load data
    print(f"[1/5] Loading {dataset_name} dataset...")
    try:
        train_ds = get_dataset(dataset_name, 'train', seq_len=seq_len)
        val_ds = get_dataset(dataset_name, 'val', seq_len=seq_len)
        test_ds = get_dataset(dataset_name, 'test', seq_len=seq_len)
    except FileNotFoundError as e:
        print(f"  ⚠ Skipping: {e}")
        return None
    except Exception as e:
        print(f"  ⚠ Error loading dataset: {e}")
        traceback.print_exc()
        return None
    
    if len(train_ds) == 0 or len(test_ds) == 0:
        print(f"  Skipping: Dataset is empty. Please download {dataset_name} first.")
        return None
    
    # Apply max_samples limit for faster training
    # Use STRIDED sampling (every k-th) to maintain temporal class distribution
    max_samples = config.get('_max_samples', None)
    if max_samples and len(train_ds) > max_samples:
        val_limit = max(max_samples // 8, 1000)
        test_limit = max(max_samples // 4, 2000)
        print(f"  Limiting (strided): train {len(train_ds)} -> {max_samples}, "
              f"val {len(val_ds)} -> {min(len(val_ds), val_limit)}, "
              f"test {len(test_ds)} -> {min(len(test_ds), test_limit)}")
        # Strided indices: pick evenly-spaced samples across full range
        train_idx = np.linspace(0, len(train_ds) - 1, max_samples, dtype=int)
        val_idx = np.linspace(0, len(val_ds) - 1, min(len(val_ds), val_limit), dtype=int)
        test_idx = np.linspace(0, len(test_ds) - 1, min(len(test_ds), test_limit), dtype=int)
        train_ds = Subset(train_ds, train_idx.tolist())
        val_ds = Subset(val_ds, val_idx.tolist())
        test_ds = Subset(test_ds, test_idx.tolist())
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    
    # Get input dimension (handle Subset wrapper)
    base_ds = train_ds.dataset if isinstance(train_ds, Subset) else train_ds
    input_dim = base_ds.get_feature_count()
    print(f"  Features: {input_dim}, Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")
    
    # Initialize model
    print(f"[2/5] Initializing {model_name} (d_model={d_model}, n_layers={n_layers})...")
    model = get_model(model_name, input_dim=input_dim, d_model=d_model, n_layers=n_layers)
    model = model.to(device)
    
    param_count = count_params(model)
    print(f"  Parameters: {param_count:,}")
    
    # Compute class weight from FULL training set (not sampled)
    # pos_weight is computed once and kept constant throughout training
    try:
        print(f"  Computing class weights from full training set ({len(train_ds)} samples)...")
        all_labels = []
        for i in range(len(train_ds)):
            _, label = train_ds[int(i)]
            all_labels.append(float(label))
        all_labels = np.array(all_labels)
        n_pos = all_labels.sum()
        n_neg = len(all_labels) - n_pos
        if n_pos > 0 and n_neg > 0:
            pos_weight = float(n_neg / n_pos)
            # Cap to avoid extreme values on highly imbalanced sets
            pos_weight = min(pos_weight, 10.0)
        else:
            pos_weight = 1.0
        print(f"  Class balance: {n_neg:.0f} benign / {n_pos:.0f} attack (full set) -> pos_weight={pos_weight:.2f}")
    except Exception:
        pos_weight = 1.0
    
    # Train
    print(f"[3/5] Training {model_name}...")
    train_result = train_model(model, train_loader, val_loader, config, device, pos_weight=pos_weight, model_name=model_name)
    print(f"  Best Val F1: {train_result['best_val_f1']:.4f} "
          f"({train_result['epochs_trained']} epochs, {train_result['train_time_sec']:.1f}s)")
    
    # Save model checkpoint
    ckpt_dir = config.get('paths', {}).get('model_dir', 'outputs/checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, f"{dataset_name}_{model_name}_seed{seed}.pt")
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_name': model_name,
        'dataset': dataset_name,
        'seed': seed,
        'input_dim': input_dim,
        'd_model': d_model,
        'n_layers': n_layers,
        'best_val_f1': train_result['best_val_f1'],
        'epochs_trained': train_result['epochs_trained'],
        'train_time_sec': train_result['train_time_sec'],
    }, ckpt_path)
    print(f"  💾 Model saved to: {ckpt_path}")
    
    # Evaluate on test set
    print(f"[4/5] Evaluating on test set...")
    test_metrics = evaluate_model(model, test_loader, device)
    print(f"  Test Acc: {test_metrics['accuracy']:.4f} | "
          f"Prec: {test_metrics['precision']:.4f} | "
          f"Rec: {test_metrics['recall']:.4f} | "
          f"F1: {test_metrics['f1']:.4f} | "
          f"AUC: {test_metrics['auc_roc']:.4f}")
    
    # Efficiency
    print(f"[5/5] Measuring efficiency...")
    sample_input = torch.randn(1, seq_len, input_dim)
    efficiency = measure_efficiency(model, sample_input, device)
    print(f"  Latency: {efficiency['latency_ms']:.2f}ms | "
          f"Throughput: {efficiency['throughput_per_sec']:.0f}/s | "
          f"Memory: {efficiency['memory_mb']:.2f}MB")
    
    # Per-attack metrics
    try:
        test_base = test_ds.dataset if isinstance(test_ds, Subset) else test_ds
        attack_metrics = per_attack_metrics(model, test_base, test_loader, device)
        attack_types_found = list(attack_metrics.keys())
        print(f"  Attack types analyzed: {len(attack_types_found)}")
    except Exception as e:
        print(f"  Per-attack analysis skipped: {e}")
        attack_metrics = {}
    
    # Compile results (remove numpy arrays for JSON serialization)
    result = {
        'dataset': dataset_name,
        'model': model_name,
        'seed': seed,
        'param_count': param_count,
        'train': {
            'best_val_f1': train_result['best_val_f1'],
            'train_time_sec': train_result['train_time_sec'],
            'epochs_trained': train_result['epochs_trained'],
            'history': train_result['history'],  # Per-epoch val_f1, val_loss, train_loss
        },
        'test': {
            'accuracy': test_metrics['accuracy'],
            'precision': test_metrics['precision'],
            'recall': test_metrics['recall'],
            'f1': test_metrics['f1'],
            'auc_roc': test_metrics['auc_roc'],
        },
        'efficiency': efficiency,
        'per_attack': attack_metrics,
    }
    
    return result


def main():
    parser = argparse.ArgumentParser(description='Comprehensive IDS Benchmark Runner')
    parser.add_argument('--config', type=str, default='configs/experiment.yaml',
                        help='Path to experiment config')
    parser.add_argument('--datasets', nargs='+', default=None,
                        help='Datasets to test (default: all from config)')
    parser.add_argument('--models', nargs='+', default=None,
                        help='Models to test (default: all from config)')
    parser.add_argument('--seeds', nargs='+', type=int, default=None,
                        help='Random seeds (default: from config)')
    parser.add_argument('--quick', action='store_true',
                        help='Quick test mode (2 epochs, 50k max samples)')
    parser.add_argument('--max-samples', type=int, default=None,
                        help='Max training samples per dataset (for CPU speed)')
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Override from CLI
    datasets = args.datasets or config.get('datasets', ['CICIDS2017'])
    models = args.models or config.get('models', list_models())
    seeds = args.seeds or config.get('seeds', [42, 123, 456, 789, 1024])
    
    if args.quick:
        config['training']['epochs'] = 10
        config['training']['early_stopping']['patience'] = 5
        if args.max_samples is None:
            args.max_samples = 50000
        print(f"[QUICK MODE] epochs=10, patience=5, max_samples={args.max_samples}")
    
    # Store max_samples in config for use in run_single_experiment
    config['_max_samples'] = args.max_samples
    
    # Device
    device_str = config.get('device', 'auto')
    if device_str == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device_str)
    
    print("=" * 70)
    print("COMPREHENSIVE IDS BENCHMARK")
    print("=" * 70)
    print(f"Datasets:  {datasets}")
    print(f"Models:    {models}")
    print(f"Seeds:     {seeds}")
    print(f"Device:    {device}")
    print(f"Epochs:    {config['training']['epochs']}")
    print(f"Batch:     {config['training']['batch_size']}")
    print(f"Seq Len:   {config['dataset']['seq_len']}")
    print(f"d_model:   {config['model']['d_model']}")
    print(f"n_layers:  {config['model']['n_layers']}")
    print("=" * 70)
    
    # Output directory
    output_dir = config.get('paths', {}).get('output_dir', 'outputs/benchmark_results')
    os.makedirs(output_dir, exist_ok=True)
    
    # Run all experiments
    all_results = []
    total = len(datasets) * len(seeds) * len(models)
    current = 0
    
    for dataset_name in datasets:
        for seed in seeds:
            for model_name in models:
                current += 1
                print(f"\n[{current}/{total}] Running experiment...")
                
                result = run_single_experiment(
                    dataset_name, model_name, seed, config, device
                )
                
                if result is not None:
                    all_results.append(result)
                    
                    # Save incrementally
                    result_file = os.path.join(
                        output_dir,
                        f"{dataset_name}_{model_name}_seed{seed}.json"
                    )
                    with open(result_file, 'w') as f:
                        json.dump(result, f, indent=2)
                
                # Free GPU memory between experiments to prevent OOM
                import gc
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    
    # Save combined results
    combined_file = os.path.join(output_dir, 'all_results.json')
    with open(combined_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Print summary
    print("\n\n" + "=" * 70)
    print("BENCHMARK SUMMARY")
    print("=" * 70)
    
    if not all_results:
        print("No results generated. Please ensure datasets are downloaded.")
        print("\nDataset download instructions:")
        print("  CICIDS2017: https://www.unb.ca/cic/datasets/ids-2017.html")
        print("  UNSW-NB15:  https://research.unsw.edu.au/projects/unsw-nb15-dataset")
        print("  CIC-IDS2018: aws s3 sync --no-sign-request --region ap-south-1 \\")
        print('    "s3://cse-cic-ids2018/Processed Traffic Data for ML Algorithms/" \\')
        print("    data/raw/CIC-IDS2018/")
        return
    
    # Group results by dataset
    from collections import defaultdict
    by_dataset = defaultdict(lambda: defaultdict(list))
    
    for r in all_results:
        by_dataset[r['dataset']][r['model']].append(r)
    
    for ds_name, models_dict in by_dataset.items():
        print(f"\n📊 {ds_name}")
        print(f"{'Model':<15} {'Acc':>8} {'Prec':>8} {'Rec':>8} {'F1':>8} {'AUC':>8} {'Lat(ms)':>8} {'Params':>10}")
        print("-" * 83)
        
        for model_name, results in models_dict.items():
            accs = [r['test']['accuracy'] for r in results]
            precs = [r['test']['precision'] for r in results]
            recs = [r['test']['recall'] for r in results]
            f1s = [r['test']['f1'] for r in results]
            aucs = [r['test']['auc_roc'] for r in results]
            lats = [r['efficiency']['latency_ms'] for r in results]
            params = results[0]['param_count']
            
            print(f"{model_name:<15} "
                  f"{np.mean(accs):.4f}  "
                  f"{np.mean(precs):.4f}  "
                  f"{np.mean(recs):.4f}  "
                  f"{np.mean(f1s):.4f}  "
                  f"{np.mean(aucs):.4f}  "
                  f"{np.mean(lats):6.2f}  "
                  f"{params:>10,}")
    
    print(f"\n✅ Results saved to: {output_dir}/")
    print(f"   Run 'python generate_tables.py' to generate publication tables.")


if __name__ == '__main__':
    main()
