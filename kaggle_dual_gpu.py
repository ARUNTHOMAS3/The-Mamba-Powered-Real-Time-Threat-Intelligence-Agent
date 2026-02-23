#!/usr/bin/env python
"""
Dual-GPU Parallel Benchmark Runner
====================================
Splits experiments across 2 GPUs for ~2x speedup on Kaggle T4 x2.

Usage:
    python kaggle_dual_gpu.py
"""

import subprocess
import sys
import os
import torch
import time


def main():
    n_gpus = torch.cuda.device_count()
    print(f"🔍 Detected {n_gpus} GPU(s):")
    for i in range(n_gpus):
        name = torch.cuda.get_device_name(i)
        vram = torch.cuda.get_device_properties(i).total_memory / 1e9
        print(f"   GPU {i}: {name} ({vram:.1f} GB)")
    
    if n_gpus < 2:
        print("\n⚠ Only 1 GPU detected. Running normally on GPU 0...")
        os.execv(sys.executable, [sys.executable, "-u", "run_benchmark.py",
                                   "--datasets", "CICIDS2017", "UNSW-NB15",
                                   "--seeds", "42", "123", "456",
                                   "--max_samples", "200000"])
        return
    
    print(f"\n🚀 Running on {n_gpus} GPUs in parallel!")
    print("   GPU 0: Seeds [42, 123]  — CICIDS2017 + UNSW-NB15 (200K samples)")
    print("   GPU 1: Seed  [456]      — CICIDS2017 + UNSW-NB15 (200K samples)")
    print("=" * 60)
    
    import threading
    
    start = time.time()
    
    env0 = os.environ.copy()
    env0["CUDA_VISIBLE_DEVICES"] = "0"
    
    env1 = os.environ.copy()
    env1["CUDA_VISIBLE_DEVICES"] = "1"
    
    def stream_output(proc, gpu_id):
        """Read process output line by line and print with GPU prefix."""
        for line in iter(proc.stdout.readline, ''):
            print(f"[GPU {gpu_id}] {line}", end='', flush=True)
    
    # Start both processes with live output
    p0 = subprocess.Popen(
        [sys.executable, "-u", "run_benchmark.py",
         "--datasets", "CICIDS2017", "UNSW-NB15",
         "--seeds", "42", "123",
         "--max_samples", "200000"],
        env=env0,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1
    )
    print(f"✅ GPU 0 started (PID {p0.pid}) — seeds [42, 123]")
    
    p1 = subprocess.Popen(
        [sys.executable, "-u", "run_benchmark.py",
         "--datasets", "CICIDS2017", "UNSW-NB15",
         "--seeds", "456",
         "--max_samples", "200000"],
        env=env1,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1
    )
    print(f"✅ GPU 1 started (PID {p1.pid}) — seed [456]")
    
    print("\n⏳ Both GPUs training in parallel — live output below:\n")
    
    # Stream output from both GPUs in parallel threads
    t0 = threading.Thread(target=stream_output, args=(p0, 0), daemon=True)
    t1 = threading.Thread(target=stream_output, args=(p1, 1), daemon=True)
    t0.start()
    t1.start()
    
    # Wait for both
    r0 = p0.wait()
    r1 = p1.wait()
    t0.join()
    t1.join()
    
    elapsed = time.time() - start
    print(f"\n✅ GPU 0 finished (exit={r0})")
    print(f"✅ GPU 1 finished (exit={r1})")
    print(f"⏱️  Total time: {int(elapsed//3600)}h {int((elapsed%3600)//60)}m")
    
    total = time.time() - start
    hours = int(total // 3600)
    mins = int((total % 3600) // 60)
    print(f"\n{'='*60}")
    print(f"🏁 All done! Total time: {hours}h {mins}m")
    print(f"{'='*60}")
    
    # Merge results
    print("\n📊 Merging results from both GPUs...")
    merge_results()
    
    # Generate tables
    print("📋 Generating publication tables...")
    subprocess.run([sys.executable, "generate_tables.py"])
    print("\n✅ Results saved to outputs/benchmark_results/")


def merge_results():
    """Merge individual JSON result files from both GPU runs into all_results.json."""
    import json
    import glob
    
    results_dir = "outputs/benchmark_results"
    pattern = os.path.join(results_dir, "*_seed*.json")
    result_files = sorted(glob.glob(pattern))
    
    if not result_files:
        print("   ⚠ No individual result files found in outputs/benchmark_results/")
        return
    
    all_results = []
    for f in result_files:
        try:
            with open(f, 'r') as fh:
                result = json.load(fh)
                all_results.append(result)
        except Exception as e:
            print(f"   ⚠ Could not load {f}: {e}")
    
    # Save combined results
    combined_path = os.path.join(results_dir, "all_results.json")
    with open(combined_path, 'w') as fh:
        json.dump(all_results, fh, indent=2)
    
    print(f"   ✅ Merged {len(all_results)} experiments from {len(result_files)} files into all_results.json")


if __name__ == '__main__':
    main()
