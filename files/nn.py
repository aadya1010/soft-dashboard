"""
nn.py -- FPS Predictor using the custom C++ Deep Learning backend.

Loads training data from fps_data.csv, trains a 2-layer neural network,
saves the trained model state to disk, and runs inference demos.

Engine workarounds (required by current backprop_b.cpp):
  1. .mul_naive()  -- optimized matmul has no backward pass
  2. X.relu()      -- converts leaf input into intermediate node for gradient flow
  3. sc_tensor_mse_loss C-API call -- bypasses eager-mode segfault
  4. No biases     -- add_bias backward is unimplemented
"""

import os
import numpy as np
from tensor import MemoryPool, Tensor, ComputationGraph
from wrapper import SCBackend, sc_tensor_mse_loss


# ============================================================================
# 1. Configuration
# ============================================================================

SCRIPT_DIR     = os.path.dirname(os.path.abspath(__file__))
CSV_PATH       = os.path.join(SCRIPT_DIR, "fps_data.csv")
MODEL_PATH     = os.path.join(SCRIPT_DIR, "model_state.npz")
HIDDEN_DIM     = 16
EPOCHS         = 500
LR             = 0.05

POOL_SIZE      = 50 * 1024 * 1024
META_POOL_SIZE = 10 * 1024 * 1024
GRAD_POOL_SIZE = 50 * 1024 * 1024


# ============================================================================
# 2. Data Loading
# ============================================================================

def load_csv(path: str):
    """
    Load fps_data.csv and return (X, Y) as float32 NumPy arrays.

    Expects a header row followed by numeric data.
    Last column is the target (FPS); all preceding columns are features.

    Returns
    -------
    X : np.ndarray, shape (N, F) -- feature matrix
    Y : np.ndarray, shape (N, 1) -- target column
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Dataset not found at '{path}'.\n"
            "Run `python prepare_data.py` first to create fps_data.csv."
        )

    data = np.loadtxt(path, delimiter=",", skiprows=1, dtype=np.float32)
    X = data[:, :-1]
    Y = data[:,  -1:]
    return X, Y


# ============================================================================
# 3. Normalization Utilities
# ============================================================================

def compute_norm_params(arr: np.ndarray, axis: int = 0):
    """Compute per-column (or global) min and max for min-max scaling."""
    return arr.min(axis=axis, keepdims=True), arr.max(axis=axis, keepdims=True)


def normalize(arr: np.ndarray, vmin: np.ndarray, vmax: np.ndarray) -> np.ndarray:
    """Scale values to [0, 1] using precomputed min/max."""
    return (arr - vmin) / (vmax - vmin + 1e-8)


def denormalize(arr: np.ndarray, vmin: np.ndarray, vmax: np.ndarray) -> np.ndarray:
    """Inverse-transform from [0, 1] back to original scale."""
    return arr * (vmax - vmin + 1e-8) + vmin


# ============================================================================
# 4. Inference Function
# ============================================================================

def predict_fps(user_specs, min_specs, rec_specs,
                x_min, x_max, y_min, y_max,
                W1, W2, pool):
    """
    Predict FPS for a single hardware profile.

    Parameters
    ----------
    user_specs : list/array of 3 -- [cpu_ghz, ram_gb, vram_gb]
    min_specs  : list/array of 3 -- game minimum requirements
    rec_specs  : list/array of 3 -- game recommended requirements
    x_min, x_max : np.ndarray    -- saved feature normalization params (shape 1xF)
    y_min, y_max : np.ndarray    -- saved target normalization params
    W1, W2     : Tensor          -- trained weight tensors
    pool       : MemoryPool      -- memory pool for tensor allocation

    Returns
    -------
    float -- predicted FPS on the original (de-normalized) scale
    """
    raw = np.array([list(user_specs) + list(min_specs) + list(rec_specs)],
                   dtype=np.float32)

    normed = normalize(raw, x_min, x_max)

    X_test     = Tensor.from_numpy(pool, normed)
    X_test_act = X_test.relu()
    hidden     = X_test_act.mul_naive(W1).relu()
    pred       = hidden.mul_naive(W2)

    pred_normed = pred.numpy()[0][0]
    return float(denormalize(np.array([[pred_normed]]), y_min, y_max)[0][0])


# ============================================================================
# 5. Model Persistence
# ============================================================================

def save_model(path, W1, W2, x_min, x_max, y_min, y_max,
               final_loss=0.0, num_samples=0):
    """Save trained weights, normalization params, and training metadata."""
    np.savez(
        path,
        W1=W1.numpy(),
        W2=W2.numpy(),
        x_min=x_min,
        x_max=x_max,
        y_min=y_min,
        y_max=y_max,
        final_loss=np.float64(final_loss),
        num_samples=np.int64(num_samples),
    )


def load_model(path):
    """
    Load model state from an .npz file.

    Returns
    -------
    dict with keys: W1, W2, x_min, x_max, y_min, y_max (all np.ndarray)
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Model file not found at '{path}'.")
    data = np.load(path)
    return {
        "W1":          data["W1"],
        "W2":          data["W2"],
        "x_min":       data["x_min"],
        "x_max":       data["x_max"],
        "y_min":       data["y_min"],
        "y_max":       data["y_max"],
        "final_loss":  float(data["final_loss"]) if "final_loss" in data else 0.0,
        "num_samples": int(data["num_samples"]) if "num_samples" in data else 0,
    }


# ============================================================================
# 6. Main Training Pipeline
# ============================================================================

def main():
    print("FPS Predictor -- Neural Network Training")
    print("-" * 50)

    # -- Memory Pools --
    pool_cpu  = MemoryPool(capacity_bytes=POOL_SIZE)
    meta_pool = MemoryPool(capacity_bytes=META_POOL_SIZE)
    grad_pool = MemoryPool(capacity_bytes=GRAD_POOL_SIZE)

    # -- Load Dataset --
    X_raw, Y_raw = load_csv(CSV_PATH)
    BATCH_SIZE, FEATURES = X_raw.shape

    print(f"[DATA]  Loaded {BATCH_SIZE} samples x {FEATURES} features "
          f"from {os.path.basename(CSV_PATH)}")

    # -- Normalize --
    x_min, x_max = compute_norm_params(X_raw, axis=0)
    y_min, y_max = compute_norm_params(Y_raw, axis=0)

    X_np = normalize(X_raw, x_min, x_max).astype(np.float32)
    Y_np = normalize(Y_raw, y_min, y_max).astype(np.float32)

    # -- Create Tensors --
    X        = Tensor.from_numpy(pool_cpu, X_np, requires_grad=True)
    Y_target = Tensor.from_numpy(pool_cpu, Y_np, requires_grad=True)

    # -- Model Weights (no biases) --
    W1 = Tensor.random_normal(pool_cpu, shape=[FEATURES, HIDDEN_DIM],
                              std=0.1, requires_grad=True)
    W2 = Tensor.random_normal(pool_cpu, shape=[HIDDEN_DIM, 1],
                              std=0.1, requires_grad=True)

    print(f"[MODEL] Architecture: {FEATURES} -> {HIDDEN_DIM} -> 1  (no biases)")

    # -- Forward Pass --
    X_act       = X.relu()
    hidden      = X_act.mul_naive(W1).relu()
    predictions = hidden.mul_naive(W2)

    loss_handle = sc_tensor_mse_loss(pool_cpu.handle, predictions.handle,
                                     Y_target.handle)
    loss_tensor = Tensor(pool_cpu, loss_handle, requires_grad=True)

    # -- Build Computation Graph --
    graph = ComputationGraph.build(
        meta_pool=meta_pool,
        loss=loss_tensor,
        pool_gpu=None,
        pool_grad_cpu=grad_pool,
        pool_grad_gpu=None,
        backend=SCBackend.CPU,
    )
    print(f"[GRAPH] Computation graph built -- {graph.size} nodes")

    # -- Training Loop --
    print(f"\n[TRAIN] Starting {EPOCHS} epochs, LR={LR}\n")

    for epoch in range(EPOCHS):
        graph.step(learning_rate=LR)

        if epoch % 50 == 0 or epoch == EPOCHS - 1:
            print(f"  Epoch {epoch:04d} | Loss: {graph.loss:.6f}")

    print("\n[TRAIN] Training complete.")

    # -- Save Model --
    final_loss = graph.loss
    save_model(MODEL_PATH, W1, W2, x_min, x_max, y_min, y_max,
               final_loss=final_loss, num_samples=BATCH_SIZE)
    print(f"[SAVE]  Model saved to {os.path.basename(MODEL_PATH)}")
    print(f"        Final loss: {final_loss:.6f} | Samples: {BATCH_SIZE}")

    # -- Inference Demo --
    print("\n" + "=" * 55)
    print("  INFERENCE -- Predicting FPS for sample hardware")
    print("=" * 55)

    test_cases = [
        {
            "label": "Mid-Range (i5 14th Gen + RTX 4060)",
            "user": [4.5, 16.0, 8.0],
            "min":  [3.3,  8.0, 3.0],
            "rec":  [3.6, 16.0, 6.0],
        },
        {
            "label": "High-End (i7 14th Gen + RTX 4090)",
            "user": [5.2, 32.0, 24.0],
            "min":  [2.0,  8.0,  4.0],
            "rec":  [3.5, 16.0,  8.0],
        },
        {
            "label": "Budget (Ryzen 5 + GTX 1650)",
            "user": [3.2, 8.0, 4.0],
            "min":  [2.5, 8.0, 2.0],
            "rec":  [3.8, 16.0, 6.0],
        },
    ]

    for tc in test_cases:
        fps = predict_fps(
            user_specs=tc["user"],
            min_specs=tc["min"],
            rec_specs=tc["rec"],
            x_min=x_min, x_max=x_max,
            y_min=y_min, y_max=y_max,
            W1=W1, W2=W2, pool=pool_cpu,
        )
        print(f"\n  {tc['label']}")
        print(f"    User -> CPU: {tc['user'][0]} GHz | "
              f"RAM: {tc['user'][1]} GB | VRAM: {tc['user'][2]} GB")
        print(f"    Predicted FPS: {fps:.2f}")

    print("\n" + "=" * 55)


if __name__ == "__main__":
    main()