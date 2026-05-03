"""
generate_dataset.py — Creates a realistic fps_data.csv for training.

Simulates a plausible FPS based on how much a user's PC exceeds the
game's recommended specs.  Pure synthetic data, but the relationship
is non-trivial enough to require a neural network to approximate.
"""

import numpy as np
import csv

np.random.seed(42)
N = 200  # number of samples

# --- Feature generation (all strictly positive) ---
user_cpu  = np.random.uniform(2.0, 5.5,  N)
user_ram  = np.random.uniform(8.0, 32.0, N)
user_vram = np.random.uniform(4.0, 24.0, N)

min_cpu   = np.random.uniform(1.5, 3.0, N)
min_ram   = np.random.uniform(4.0, 8.0, N)
min_vram  = np.random.uniform(2.0, 4.0, N)

rec_cpu   = np.random.uniform(3.0, 4.5,  N)
rec_ram   = np.random.uniform(8.0, 16.0, N)
rec_vram  = np.random.uniform(6.0, 12.0, N)

# --- Synthetic FPS formula ---
# Headroom = how much the user exceeds recommended specs (clamped ≥ 0)
cpu_headroom  = np.maximum(user_cpu  - rec_cpu,  0)
ram_headroom  = np.maximum(user_ram  - rec_ram,  0)
vram_headroom = np.maximum(user_vram - rec_vram, 0)

# Base FPS from meeting minimum specs + bonus from headroom
base_fps = 30 + 20 * (user_cpu / rec_cpu)
fps = (
    base_fps
    + 15 * cpu_headroom
    + 1.5 * ram_headroom
    + 4.0 * vram_headroom
    + np.random.normal(0, 3, N)  # noise
)
fps = np.clip(fps, 30, 165)

# --- Write CSV ---
header = [
    "user_cpu", "user_ram", "user_vram",
    "min_cpu",  "min_ram",  "min_vram",
    "rec_cpu",  "rec_ram",  "rec_vram",
    "fps",
]

with open("fps_data.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(header)
    for i in range(N):
        writer.writerow([
            f"{user_cpu[i]:.4f}",  f"{user_ram[i]:.4f}",  f"{user_vram[i]:.4f}",
            f"{min_cpu[i]:.4f}",   f"{min_ram[i]:.4f}",   f"{min_vram[i]:.4f}",
            f"{rec_cpu[i]:.4f}",   f"{rec_ram[i]:.4f}",   f"{rec_vram[i]:.4f}",
            f"{fps[i]:.4f}",
        ])

print(f"[OK] Generated fps_data.csv with {N} samples and {len(header)} columns.")
