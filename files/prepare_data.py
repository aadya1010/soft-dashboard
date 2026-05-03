"""
prepare_data.py — Merges raw FPS benchmark data with game requirements
to produce the clean fps_data.csv expected by nn.py.

Input  (from dataset/):
    fps_benchmark.csv          — 24K rows of (CPU, GPU, game, resolution, setting, FPS)
    videogame_requirements.csv — 10K+ rows of (game, min/rec CPU, RAM, VRAM, …)

Output (to files/):
    fps_data.csv — Clean 10-column CSV (9 features + 1 target):
        user_cpu, user_ram, user_vram,
        min_cpu, min_ram, min_vram,
        rec_cpu, rec_ram, rec_vram,
        fps

All values are float32-compatible (GHz for CPUs, GB for memory, raw for FPS).
"""

import csv
import os
import re
import sys

# ============================================================================
# Paths
# ============================================================================

SCRIPT_DIR    = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR   = os.path.join(SCRIPT_DIR, "dataset")
FPS_CSV       = os.path.join(DATASET_DIR, "fps_benchmark.csv")
REQS_CSV      = os.path.join(DATASET_DIR, "videogame_requirements.csv")
OUTPUT_CSV    = os.path.join(SCRIPT_DIR, "fps_data.csv")

# ============================================================================
# Game Name Mapping
# ============================================================================
# FPS benchmark uses camelCase names wrapped in b'…'.
# Requirements CSV uses natural English titles.
# This hand-verified mapping links the 24 benchmark games to the best
# matching base-game entry in the requirements file.

GAME_MAP = {
    "aWayOut":                        "A Way Out",
    "airMechStrike":                  "AirMech",
    "apexLegends":                    "Apex Legends",
    "battlefield4":                   "Battlefield 4: Second Assault",
    "battletech":                     "BattleTech",
    "callOfDutyWW2":                  "Call of Duty: WWII",
    "counterStrikeGlobalOffensive":   "Counter-Strike: Global Offensive",
    "destiny2":                       "Destiny 2",
    "dota2":                          "DOTA 2",
    "farCry5":                        "Far Cry 5",
    "fortnite":                       "Fortnite",
    "frostpunk":                      "Frostpunk",
    "grandTheftAuto5":                "Grand Theft Auto V",
    "leagueOfLegends":                "League of Legends",
    "overwatch":                      "Overwatch",
    "pathOfExile":                    "Path of Exile",
    "playerUnknownsBattlegrounds":    "PlayerUnknowns Battlegrounds",
    "radicalHeights":                 "Radical Heights",
    "rainbowSixSiege":                "Rainbow Six: Siege",
    "seaOfThieves":                   "Sea of Thieves",
    "starcraft2":                     "StarCraft II: Wings of Liberty",
    "totalWar3Kingdoms":              "Total War: Three Kingdoms",
    "warframe":                       "Warframe",
    "worldOfTanks":                   "World of Tanks",
}


# ============================================================================
# Parsing Helpers
# ============================================================================

def clean_bytes_string(val: str) -> str:
    """Strip the b'…' wrapper that some columns carry from a prior pickle export."""
    val = val.strip()
    if val.startswith("b'") and val.endswith("'"):
        return val[2:-1]
    if val.startswith('b"') and val.endswith('"'):
        return val[2:-1]
    return val


def parse_memory_gb(val: str) -> float:
    """
    Parse a memory string into GB as a float.

    Handles formats:
        "8 GB", "8GB", "512MB", "512 MB", "1.953125GB",
        "0", "0MB", "", None  →  NaN
    """
    if val is None:
        return float("nan")
    val = val.strip().upper()
    if not val or val in ("0", "0MB", "0 MB", "0GB", "0 GB"):
        return float("nan")

    # Try "<number> GB" or "<number>GB"
    m = re.match(r"^([\d.]+)\s*GB$", val)
    if m:
        return float(m.group(1))

    # Try "<number> MB" or "<number>MB"  →  convert to GB
    m = re.match(r"^([\d.]+)\s*MB$", val)
    if m:
        return float(m.group(1)) / 1024.0

    # Last resort: try bare number (assume GB)
    try:
        v = float(val)
        return v if v > 0 else float("nan")
    except ValueError:
        return float("nan")


def safe_float(val: str, default=float("nan")) -> float:
    """Convert string to float, returning default on failure."""
    try:
        v = float(val)
        return v if v > 0 else default
    except (ValueError, TypeError):
        return default


# ============================================================================
# Step 1 — Load Game Requirements
# ============================================================================

def load_requirements(path: str) -> dict:
    """
    Returns a dict mapping game name → {
        min_cpu, rec_cpu,  (GHz, float)
        min_ram, rec_ram,  (GB, float — may be NaN)
        min_vram, rec_vram (GB, float — may be NaN)
    }
    """
    reqs = {}
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f)
        header = next(reader)

        # Column indices (verified from header inspection)
        i_name      = 0
        i_min_cpu   = header.index("Min_CPU_CPU_Speed")        # 1
        i_rec_cpu   = header.index("Recom_CPU_CPU_Speed")      # 16
        i_min_ram   = header.index("Min_RAM")                  # 79
        i_rec_ram   = header.index("Recom_RAM")                # 80
        i_min_vram  = header.index("Min_VRAM")                 # 81
        i_rec_vram  = header.index("Recom_VRAM")               # 82

        for row in reader:
            if len(row) <= max(i_rec_vram, i_min_vram, i_rec_ram):
                continue

            name = row[i_name].strip()
            reqs[name] = {
                "min_cpu":  safe_float(row[i_min_cpu]),
                "rec_cpu":  safe_float(row[i_rec_cpu]),
                "min_ram":  parse_memory_gb(row[i_min_ram]),
                "rec_ram":  parse_memory_gb(row[i_rec_ram]),
                "min_vram": parse_memory_gb(row[i_min_vram]),
                "rec_vram": parse_memory_gb(row[i_rec_vram]),
            }

    return reqs


# ============================================================================
# Step 2 — Load FPS Benchmark & Merge
# ============================================================================

def load_and_merge(fps_path: str, reqs: dict) -> list:
    """
    Read every row in fps_benchmark.csv, extract user-PC features,
    look up the game in the requirements dict, and assemble the final
    10-column feature row.

    Returns a list of dicts, one per valid merged row.
    """
    merged_rows = []
    stats = {"total": 0, "no_map": 0, "no_req": 0, "bad_parse": 0, "nan_dropped": 0}

    with open(fps_path, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f)
        header = next(reader)

        # FPS benchmark column indices
        i_cpu_freq   = header.index("CpuFrequency")     # MHz → /1000 = GHz
        i_gpu_mem    = header.index("GpuMemorySize")     # MB  → /1000 = GB
        i_cpu_cache  = header.index("CpuCacheL3")        # MB  (proxy for user RAM tier)
        i_game_name  = header.index("GameName")
        i_fps        = header.index("FPS")

        # The benchmark CSV doesn't have explicit "user RAM" in GB.
        # We'll derive a proxy: systems with bigger L3 caches tend to ship
        # with more RAM. But a more reliable proxy is CpuNumberOfThreads × 2
        # (consumer platforms: 8T → 16 GB, 16T → 32 GB, 4T → 8 GB).
        i_cpu_threads = header.index("CpuNumberOfThreads")

        for row in reader:
            stats["total"] += 1

            # --- Parse game name (strip b'…' wrapper) ---
            game_raw = clean_bytes_string(row[i_game_name])
            if game_raw not in GAME_MAP:
                stats["no_map"] += 1
                continue

            req_name = GAME_MAP[game_raw]
            if req_name not in reqs:
                stats["no_req"] += 1
                continue

            req = reqs[req_name]

            # --- User PC features ---
            user_cpu  = safe_float(row[i_cpu_freq]) / 1000.0    # MHz → GHz
            user_vram = safe_float(row[i_gpu_mem])  / 1000.0    # MB  → GB

            # User RAM proxy: threads × 2 GB (realistic consumer mapping)
            threads   = safe_float(row[i_cpu_threads])
            user_ram  = threads * 2.0 if threads == threads else float("nan")

            fps_val   = safe_float(row[i_fps])

            # --- Assemble the 10-value row ---
            record = {
                "user_cpu":  user_cpu,
                "user_ram":  user_ram,
                "user_vram": user_vram,
                "min_cpu":   req["min_cpu"],
                "min_ram":   req["min_ram"],
                "min_vram":  req["min_vram"],
                "rec_cpu":   req["rec_cpu"],
                "rec_ram":   req["rec_ram"],
                "rec_vram":  req["rec_vram"],
                "fps":       fps_val,
            }

            # --- Drop rows with any NaN ---
            has_nan = False
            for k, v in record.items():
                if v != v:  # NaN check
                    has_nan = True
                    break

            if has_nan:
                stats["nan_dropped"] += 1
                continue

            # --- Sanity: all values must be strictly positive (relu requirement) ---
            if any(v <= 0 for v in record.values()):
                stats["bad_parse"] += 1
                continue

            merged_rows.append(record)

    return merged_rows, stats


# ============================================================================
# Step 3 — Impute Missing VRAM via Per-Game Median
# ============================================================================
# Many games have VRAM=0 in the requirements CSV. Instead of dropping them
# (which would lose most of the dataset), we can run a two-pass strategy:
#   Pass 1: merge with NaN allowed for VRAM only, collect per-game VRAM values
#   Pass 2: fill NaN VRAM with median from games that DO have VRAM data
# But since Step 2 already drops NaN rows and the remaining 10+ games with
# valid VRAM still give us thousands of rows, we keep the strict approach.


# ============================================================================
# Step 4 — Write Output CSV
# ============================================================================

def write_csv(rows: list, path: str):
    """Write the list of dicts to a clean CSV file."""
    columns = [
        "user_cpu", "user_ram", "user_vram",
        "min_cpu",  "min_ram",  "min_vram",
        "rec_cpu",  "rec_ram",  "rec_vram",
        "fps",
    ]

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        for row in rows:
            writer.writerow([f"{row[c]:.6f}" for c in columns])

    return len(rows)


# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 65)
    print("  prepare_data.py — Building fps_data.csv from raw datasets")
    print("=" * 65)

    # Validate input files exist
    for p, label in [(FPS_CSV, "fps_benchmark.csv"), (REQS_CSV, "videogame_requirements.csv")]:
        if not os.path.isfile(p):
            print(f"\n[ERROR] {label} not found at {p}")
            sys.exit(1)

    # --- Load requirements ---
    print(f"\n[LOAD] Loading game requirements from {os.path.basename(REQS_CSV)}...")
    reqs = load_requirements(REQS_CSV)
    print(f"   -> {len(reqs)} games loaded")

    # --- Check which mapped games actually exist ---
    mapped_found = 0
    mapped_missing = []
    for camel, title in GAME_MAP.items():
        if title in reqs:
            mapped_found += 1
        else:
            mapped_missing.append(f"  {camel} -> {title}")

    print(f"   -> {mapped_found}/{len(GAME_MAP)} benchmark games matched")
    if mapped_missing:
        print(f"   [WARN] Missing from requirements ({len(mapped_missing)}):")
        for m in mapped_missing:
            print(f"     {m}")

    # --- Merge ---
    print(f"\n[LOAD] Loading FPS benchmarks from {os.path.basename(FPS_CSV)}...")
    rows, stats = load_and_merge(FPS_CSV, reqs)
    print(f"   -> {stats['total']} total benchmark rows")
    print(f"   -> {stats['no_map']} skipped (game not in mapping)")
    print(f"   -> {stats['no_req']} skipped (game not in requirements)")
    print(f"   -> {stats['nan_dropped']} dropped (NaN in features)")
    print(f"   -> {stats['bad_parse']} dropped (non-positive values)")
    total_dropped = stats["no_map"] + stats["no_req"] + stats["nan_dropped"] + stats["bad_parse"]
    print(f"   -> {total_dropped} total rows dropped")
    print(f"   [OK] {len(rows)} clean rows remaining")

    if not rows:
        print("\n[ERROR] No valid rows after merge. Check game name mapping.")
        sys.exit(1)

    # --- Write output ---
    n = write_csv(rows, OUTPUT_CSV)
    print(f"\n[SAVE] Saved {n} rows to {os.path.basename(OUTPUT_CSV)}")

    # ==================================================================
    # Verification Summary
    # ==================================================================
    print("\n" + "=" * 65)
    print("  DATA VERIFICATION")
    print("=" * 65)

    # Re-read and verify
    import numpy as np

    data = np.loadtxt(OUTPUT_CSV, delimiter=",", skiprows=1, dtype=np.float64)
    col_names = [
        "user_cpu", "user_ram", "user_vram",
        "min_cpu",  "min_ram",  "min_vram",
        "rec_cpu",  "rec_ram",  "rec_vram",
        "fps",
    ]

    # --- df.info() equivalent ---
    print(f"\n  Shape: ({data.shape[0]}, {data.shape[1]})")
    print(f"  Dtype: float64 (all columns)")
    print(f"  Memory: {data.nbytes / 1024:.1f} KB\n")
    print(f"  {'Column':<15s} {'Non-Null':>10s} {'Min':>10s} {'Max':>10s} {'Mean':>10s}")
    print(f"  {'─' * 15}  {'─' * 10} {'─' * 10} {'─' * 10} {'─' * 10}")
    nan_total = 0
    for i, name in enumerate(col_names):
        col = data[:, i]
        nans = int(np.isnan(col).sum())
        nan_total += nans
        non_null = len(col) - nans
        print(f"  {name:<15s} {non_null:>10d} {col.min():>10.4f} {col.max():>10.4f} {col.mean():>10.4f}")

    print(f"\n  Total NaN cells: {nan_total}")

    # --- df.head() equivalent ---
    print(f"\n  {'─' * 65}")
    print(f"  First 5 rows (df.head()):")
    print(f"  {'─' * 65}")
    header_str = "  " + " ".join(f"{n:>10s}" for n in col_names)
    print(header_str)
    for i in range(min(5, len(data))):
        row_str = "  " + " ".join(f"{data[i, j]:>10.4f}" for j in range(len(col_names)))
        print(row_str)

    # --- Sanity checks ---
    print(f"\n  {'─' * 65}")
    print(f"  Sanity Checks:")
    all_positive = (data > 0).all()
    print(f"    All values > 0:     {'PASS' if all_positive else 'FAIL'}")
    ram_ok = data[:, 1].max() <= 128  # user_ram should be in GB, not MB
    print(f"    RAM in GB (<=128):  {'PASS' if ram_ok else 'FAIL -- values too large, check units!'}")
    fps_ok = data[:, -1].max() <= 500
    print(f"    FPS range (<=500):  {'PASS' if fps_ok else 'FAIL -- unrealistic FPS values!'}")
    no_nans = nan_total == 0
    print(f"    Zero NaN cells:     {'PASS' if no_nans else 'FAIL'}")

    print(f"\n  Total rows dropped: {total_dropped}")
    print(f"  Total rows saved:   {len(rows)}")
    print("=" * 65)


if __name__ == "__main__":
    main()
