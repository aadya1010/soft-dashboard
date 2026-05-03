"""
predict.py -- Interactive FPS prediction with comprehensive CLI dashboard.

Loads saved model weights and normalization parameters from model_state.npz,
looks up game requirements from the dataset, runs inference, and displays
a full analysis including predicted FPS, error margin, playability verdict,
and hardware bottleneck warnings.

Usage:
    python predict.py
"""

import os
import sys
import csv
import re
import math
import numpy as np
from tensor import MemoryPool, Tensor


# ============================================================================
# Paths
# ============================================================================

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH  = os.path.join(SCRIPT_DIR, "model_state.npz")
REQS_CSV    = os.path.join(SCRIPT_DIR, "dataset", "videogame_requirements.csv")

POOL_SIZE   = 10 * 1024 * 1024
DASH_WIDTH  = 60


# ============================================================================
# Normalization (mirrors nn.py)
# ============================================================================

def normalize(arr, vmin, vmax):
    return (arr - vmin) / (vmax - vmin + 1e-8)

def denormalize(arr, vmin, vmax):
    return arr * (vmax - vmin + 1e-8) + vmin


# ============================================================================
# Memory Parsing (mirrors prepare_data.py)
# ============================================================================

def parse_memory_gb(val):
    """Parse a memory string like '8 GB', '512MB', '2GB' into float GB."""
    if val is None:
        return None
    val = val.strip().upper()
    if not val or val in ("0", "0MB", "0 MB", "0GB", "0 GB"):
        return None


#Hello
    m = re.match(r"^([\d.]+)\s*GB$", val)
    if m:
        return float(m.group(1))

    m = re.match(r"^([\d.]+)\s*MB$", val)
    if m:
        return float(m.group(1)) / 1024.0

    try:
        v = float(val)
        return v if v > 0 else None
    except ValueError:
        return None


def safe_float(val):
    try:
        v = float(val)
        return v if v > 0 else None
    except (ValueError, TypeError):
        return None


# ============================================================================
# Game Requirements Lookup
# ============================================================================

def load_game_requirements(path):
    """
    Load all game requirements from the CSV into a dict.

    Uses fallback logic for missing values:
      - rec_* defaults to min_* if missing
      - VRAM defaults to 0.5 GB if both min and rec are missing
      - Games without at least min_cpu and min_ram are skipped
    """
    games = {}
    DEFAULT_VRAM = 0.5

    with open(path, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f)
        header = next(reader)

        i_name     = 0
        i_min_cpu  = header.index("Min_CPU_CPU_Speed")
        i_rec_cpu  = header.index("Recom_CPU_CPU_Speed")
        i_min_ram  = header.index("Min_RAM")
        i_rec_ram  = header.index("Recom_RAM")
        i_min_vram = header.index("Min_VRAM")
        i_rec_vram = header.index("Recom_VRAM")

        for row in reader:
            if len(row) <= max(i_rec_vram, i_min_vram, i_rec_ram):
                continue

            name = row[i_name].strip()

            min_cpu  = safe_float(row[i_min_cpu])
            rec_cpu  = safe_float(row[i_rec_cpu])
            min_ram  = parse_memory_gb(row[i_min_ram])
            rec_ram  = parse_memory_gb(row[i_rec_ram])
            min_vram = parse_memory_gb(row[i_min_vram])
            rec_vram = parse_memory_gb(row[i_rec_vram])

            if min_cpu is None or min_ram is None:
                continue

            if rec_cpu is None:
                rec_cpu = min_cpu
            if rec_ram is None:
                rec_ram = min_ram
            if min_vram is None:
                min_vram = DEFAULT_VRAM
            if rec_vram is None:
                rec_vram = min_vram

            games[name.lower()] = {
                "display_name": name,
                "min_cpu":  min_cpu,
                "rec_cpu":  rec_cpu,
                "min_ram":  min_ram,
                "rec_ram":  rec_ram,
                "min_vram": min_vram,
                "rec_vram": rec_vram,
            }

    return games


def search_games(query, game_db, max_results=10):
    """Fuzzy search: return games whose name contains the query substring."""
    query_lower = query.lower().strip()
    matches = []
    for key, info in game_db.items():
        if query_lower in key:
            matches.append(info)
        if len(matches) >= max_results:
            break
    return matches


# ============================================================================
# Forward Pass (mirrors nn.py architecture)
# ============================================================================

def run_inference(user_specs, min_specs, rec_specs,
                  x_min, x_max, y_min, y_max,
                  W1_np, W2_np, pool):
    """Run a single forward pass and return the predicted FPS (de-normalized)."""
    raw = np.array([list(user_specs) + list(min_specs) + list(rec_specs)],
                   dtype=np.float32)
    normed = normalize(raw, x_min, x_max)

    X_test = Tensor.from_numpy(pool, normed)
    W1     = Tensor.from_numpy(pool, W1_np.astype(np.float32))
    W2     = Tensor.from_numpy(pool, W2_np.astype(np.float32))

    X_act  = X_test.relu()
    hidden = X_act.mul_naive(W1).relu()
    pred   = hidden.mul_naive(W2)

    pred_normed = pred.numpy()[0][0]
    return float(denormalize(np.array([[pred_normed]]), y_min, y_max)[0][0])


# ============================================================================
# Dashboard Components
# ============================================================================

def get_playability_verdict(fps):
    """Return a (label, description) tuple based on predicted FPS."""
    if fps < 30:
        return "UNPLAYABLE", "Below minimum playable threshold"
    elif fps < 60:
        return "PLAYABLE", "Console-level experience (30-59 FPS)"
    elif fps < 120:
        return "SMOOTH", "Standard PC gaming experience (60-119 FPS)"
    else:
        return "COMPETITIVE / ESPORTS READY", "120+ FPS for competitive play"


def compute_error_margin(final_loss, y_min, y_max):
    """
    Convert normalized MSE loss back to real-world FPS error margin.

    The model trains on min-max normalized targets in [0, 1].
    RMSE in normalized space = sqrt(MSE).
    Scale back: error_fps = RMSE_norm * (y_max - y_min).
    """
    if final_loss <= 0:
        return 0.0
    rmse_norm = math.sqrt(final_loss)
    y_range = float(np.asarray(y_max).flat[0]) - float(np.asarray(y_min).flat[0])
    return rmse_norm * y_range


def analyze_bottlenecks(user_specs, rec_specs):
    """
    Compare user hardware against recommended specs.

    Parameters
    ----------
    user_specs : dict with keys cpu, ram, vram
    rec_specs  : dict with keys cpu, ram, vram

    Returns
    -------
    list of warning strings (empty if no bottlenecks)
    """
    warnings = []
    labels = {
        "cpu":  ("CPU Clock", "GHz"),
        "ram":  ("System RAM", "GB"),
        "vram": ("GPU VRAM", "GB"),
    }

    for key, (label, unit) in labels.items():
        user_val = user_specs[key]
        rec_val  = rec_specs[key]
        if user_val < rec_val:
            deficit_pct = ((rec_val - user_val) / rec_val) * 100
            warnings.append(
                f"  [!] {label}: {user_val:.1f} {unit} is below "
                f"recommended {rec_val:.1f} {unit} "
                f"({deficit_pct:.0f}% deficit)"
            )

    return warnings


def print_dashboard(game, user_specs, predicted_fps, error_margin,
                    bottleneck_warnings, num_samples):
    """Print the full analysis dashboard to the terminal."""
    w = DASH_WIDTH

    print()
    print("=" * w)
    print("  FPS PREDICTION REPORT")
    print("=" * w)

    # -- Game & Hardware --
    print(f"  Game:  {game['display_name']}")
    print(f"  PC:    CPU {user_specs['cpu']:.1f} GHz | "
          f"RAM {user_specs['ram']:.0f} GB | "
          f"VRAM {user_specs['vram']:.0f} GB")
    print("-" * w)

    # -- Game Requirements --
    print("  GAME REQUIREMENTS")
    print(f"    Minimum      ->  CPU: {game['min_cpu']:.2f} GHz | "
          f"RAM: {game['min_ram']:.0f} GB | "
          f"VRAM: {game['min_vram']:.1f} GB")
    print(f"    Recommended  ->  CPU: {game['rec_cpu']:.2f} GHz | "
          f"RAM: {game['rec_ram']:.0f} GB | "
          f"VRAM: {game['rec_vram']:.1f} GB")
    print("-" * w)

    # -- Prediction --
    print("  PREDICTION")
    fps_display = max(0, predicted_fps)
    print(f"    Predicted FPS:    {fps_display:.1f}")
    if error_margin > 0:
        low = max(0, fps_display - error_margin)
        high = fps_display + error_margin
        print(f"    Error Margin:     +/- {error_margin:.1f} FPS "
              f"(range: {low:.0f} - {high:.0f})")
    if num_samples > 0:
        print(f"    Model trained on: {num_samples} samples")
    print("-" * w)

    # -- Playability Verdict --
    verdict, description = get_playability_verdict(fps_display)
    print("  PLAYABILITY VERDICT")
    print(f"    Rating:  {verdict}")
    print(f"    Detail:  {description}")
    print("-" * w)

    # -- Bottleneck Analysis --
    print("  HARDWARE BOTTLENECK ANALYSIS")
    if bottleneck_warnings:
        for warn in bottleneck_warnings:
            print(warn)
    else:
        print("  [OK] System fully meets recommended requirements.")
    print("=" * w)


# ============================================================================
# Programmatic API (for Streamlit / external callers)
# ============================================================================

# Module-level cache: loaded once on first call, reused thereafter.
_model_cache = None
_game_db_cache = None
_pool_cache = None


def _ensure_loaded():
    """Lazy-load model weights and game database into module-level cache."""
    global _model_cache, _game_db_cache, _pool_cache

    if _model_cache is None:
        if not os.path.isfile(MODEL_PATH):
            raise FileNotFoundError(
                f"Model file not found at '{MODEL_PATH}'. "
                "Run nn.py to train and save the model first."
            )
        raw = np.load(MODEL_PATH)
        _model_cache = {
            "W1":          raw["W1"],
            "W2":          raw["W2"],
            "x_min":       raw["x_min"],
            "x_max":       raw["x_max"],
            "y_min":       raw["y_min"],
            "y_max":       raw["y_max"],
            "final_loss":  float(raw["final_loss"]) if "final_loss" in raw else 0.0,
            "num_samples": int(raw["num_samples"]) if "num_samples" in raw else 0,
        }

    if _game_db_cache is None:
        if not os.path.isfile(REQS_CSV):
            raise FileNotFoundError(
                f"Game requirements CSV not found at '{REQS_CSV}'."
            )
        _game_db_cache = load_game_requirements(REQS_CSV)

    if _pool_cache is None:
        _pool_cache = MemoryPool(capacity_bytes=POOL_SIZE)


def get_game_list():
    """
    Return a sorted list of all game display names in the database.

    Useful for populating search / autocomplete widgets in the UI.
    """
    _ensure_loaded()
    return sorted(g["display_name"] for g in _game_db_cache.values())


def get_prediction_report(cpu_ghz, ram_gb, vram_gb, game_name):
    """
    Run a full prediction pipeline and return a structured report.

    Parameters
    ----------
    cpu_ghz   : float  -- User CPU clock speed in GHz.
    ram_gb    : float  -- User system RAM in GB.
    vram_gb   : float  -- User GPU VRAM in GB.
    game_name : str    -- Exact display name of the game (from get_game_list).

    Returns
    -------
    dict with keys:
        game           : dict  -- Game spec record (display_name, min_*, rec_*).
        predicted_fps  : float -- Clamped to >= 0.
        error_margin   : float -- +/- FPS derived from training loss.
        fps_low        : float -- predicted_fps - error_margin (clamped >= 0).
        fps_high       : float -- predicted_fps + error_margin.
        verdict_label  : str   -- e.g. "SMOOTH", "UNPLAYABLE".
        verdict_detail : str   -- Human-readable description.
        bottlenecks    : list[str] -- Hardware deficit warnings (may be empty).
        num_samples    : int   -- Number of training samples the model saw.
        inference_ms   : float -- Forward-pass wall-clock time in milliseconds.
    """
    import time as _time

    _ensure_loaded()
    m = _model_cache

    # -- Resolve game by display name ----------------------------------------
    key = game_name.strip().lower()
    if key not in _game_db_cache:
        raise KeyError(
            f"Game '{game_name}' not found in the database. "
            "Use get_game_list() or search_games() to find valid names."
        )
    game = _game_db_cache[key]

    # -- Inference -----------------------------------------------------------
    user_specs = [cpu_ghz, ram_gb, vram_gb]
    min_specs  = [game["min_cpu"], game["min_ram"], game["min_vram"]]
    rec_specs  = [game["rec_cpu"], game["rec_ram"], game["rec_vram"]]

    t0 = _time.perf_counter()
    predicted_fps = run_inference(
        user_specs=user_specs,
        min_specs=min_specs,
        rec_specs=rec_specs,
        x_min=m["x_min"], x_max=m["x_max"],
        y_min=m["y_min"], y_max=m["y_max"],
        W1_np=m["W1"], W2_np=m["W2"],
        pool=_pool_cache,
    )
    t1 = _time.perf_counter()

    predicted_fps = max(0.0, predicted_fps)
    error_margin  = compute_error_margin(m["final_loss"], m["y_min"], m["y_max"])

    verdict_label, verdict_detail = get_playability_verdict(predicted_fps)

    user_hw = {"cpu": cpu_ghz, "ram": ram_gb, "vram": vram_gb}
    rec_hw  = {"cpu": game["rec_cpu"], "ram": game["rec_ram"],
               "vram": game["rec_vram"]}
    bottlenecks = analyze_bottlenecks(user_hw, rec_hw)

    return {
        "game":           game,
        "predicted_fps":  predicted_fps,
        "error_margin":   error_margin,
        "fps_low":        max(0.0, predicted_fps - error_margin),
        "fps_high":       predicted_fps + error_margin,
        "verdict_label":  verdict_label,
        "verdict_detail": verdict_detail,
        "bottlenecks":    bottlenecks,
        "num_samples":    m["num_samples"],
        "inference_ms":   round((t1 - t0) * 1000.0, 4),
    }


# ============================================================================
# Interactive CLI
# ============================================================================

def prompt_float(message, low=None, high=None):
    """Prompt user for a float value with optional range validation."""
    while True:
        try:
            val = float(input(message))
            if low is not None and val < low:
                print(f"  Value must be >= {low}. Try again.")
                continue
            if high is not None and val > high:
                print(f"  Value must be <= {high}. Try again.")
                continue
            return val
        except ValueError:
            print("  Invalid number. Try again.")


def main():
    print("=" * DASH_WIDTH)
    print("  FPS Predictor -- Interactive Inference")
    print("=" * DASH_WIDTH)

    # -- Load model --
    if not os.path.isfile(MODEL_PATH):
        print(f"\n[ERROR] Model file not found: {os.path.basename(MODEL_PATH)}")
        print("Run `python nn.py` first to train and save the model.")
        sys.exit(1)

    model = np.load(MODEL_PATH)
    W1_np       = model["W1"]
    W2_np       = model["W2"]
    x_min       = model["x_min"]
    x_max       = model["x_max"]
    y_min       = model["y_min"]
    y_max       = model["y_max"]
    final_loss  = float(model["final_loss"]) if "final_loss" in model else 0.0
    num_samples = int(model["num_samples"]) if "num_samples" in model else 0

    error_margin = compute_error_margin(final_loss, y_min, y_max)

    print(f"[OK] Model loaded from {os.path.basename(MODEL_PATH)}")
    print(f"     Weights: W1 {W1_np.shape}, W2 {W2_np.shape}")
    print(f"     Training loss: {final_loss:.6f} | "
          f"Error margin: +/- {error_margin:.1f} FPS")

    # -- Load game database --
    if not os.path.isfile(REQS_CSV):
        print(f"\n[ERROR] Game requirements not found: {REQS_CSV}")
        sys.exit(1)

    game_db = load_game_requirements(REQS_CSV)
    print(f"[OK] Loaded {len(game_db)} games with specs\n")

    # -- Memory pool --
    pool = MemoryPool(capacity_bytes=POOL_SIZE)

    # -- Main loop --
    while True:
        print("-" * DASH_WIDTH)
        print("Enter your PC specs:\n")

        cpu_ghz = prompt_float("  CPU Clock Speed (GHz, e.g. 3.6): ",
                               low=0.5, high=8.0)
        ram_gb  = prompt_float("  RAM (GB, e.g. 16): ",
                               low=1.0, high=256.0)
        vram_gb = prompt_float("  GPU VRAM (GB, e.g. 8): ",
                               low=0.5, high=48.0)

        user_hw = {"cpu": cpu_ghz, "ram": ram_gb, "vram": vram_gb}

        # -- Game lookup --
        print("\nEnter game name (or part of it):")
        query = input("  > ").strip()

        if not query:
            print("  No game name entered. Skipping.\n")
            continue

        matches = search_games(query, game_db)

        if not matches:
            print(f"  No games found matching '{query}'.")
            print("  Try a different name (e.g., 'fortnite', 'gta', "
                  "'overwatch').\n")
            continue

        # Let user pick if multiple matches
        if len(matches) == 1:
            game = matches[0]
        else:
            print(f"\n  Found {len(matches)} matching games:")
            for i, g in enumerate(matches):
                print(f"    [{i + 1}] {g['display_name']}")

            while True:
                try:
                    choice = int(input(f"\n  Select [1-{len(matches)}]: "))
                    if 1 <= choice <= len(matches):
                        game = matches[choice - 1]
                        break
                    print(f"  Enter a number between 1 and {len(matches)}.")
                except ValueError:
                    print("  Invalid input. Enter a number.")

        # -- Run prediction --
        predicted_fps = run_inference(
            user_specs=[cpu_ghz, ram_gb, vram_gb],
            min_specs=[game["min_cpu"], game["min_ram"], game["min_vram"]],
            rec_specs=[game["rec_cpu"], game["rec_ram"], game["rec_vram"]],
            x_min=x_min, x_max=x_max,
            y_min=y_min, y_max=y_max,
            W1_np=W1_np, W2_np=W2_np,
            pool=pool,
        )

        # -- Bottleneck analysis --
        rec_hw = {
            "cpu":  game["rec_cpu"],
            "ram":  game["rec_ram"],
            "vram": game["rec_vram"],
        }
        bottleneck_warnings = analyze_bottlenecks(user_hw, rec_hw)

        # -- Print dashboard --
        print_dashboard(
            game=game,
            user_specs=user_hw,
            predicted_fps=predicted_fps,
            error_margin=error_margin,
            bottleneck_warnings=bottleneck_warnings,
            num_samples=num_samples,
        )

        # -- Continue? --
        print()
        again = input("Run another prediction? [Y/n]: ").strip().lower()
        if again in ("n", "no", "q", "quit", "exit"):
            break
        print()

    print("\nDone.")


if __name__ == "__main__":
    main()

