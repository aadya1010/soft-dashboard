"""
core_inference.py
-----------------
Decoupled real-time inference backend for the Streamlit dashboard.

Loads trained model weights once at module level and exposes a single
`run_inference()` function that accepts raw feature inputs, normalizes
them, executes a forward pass through the custom C++ tensor engine,
and returns a prediction with high-precision timing.
"""

import os
import sys
import time
import numpy as np

# Ensure the files/ directory is on the import path so the C++ tensor
# module can be located regardless of the working directory.
_FILES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "files")
if _FILES_DIR not in sys.path:
    sys.path.insert(0, _FILES_DIR)

from tensor import MemoryPool, Tensor

# ============================================================================
# Global Model Loading
# ============================================================================
# Load the trained model state once at import time to avoid repeated disk I/O
# on every inference request.

_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "files", "model_state.npz"
)

if not os.path.isfile(_MODEL_PATH):
    raise FileNotFoundError(
        f"Trained model not found at '{_MODEL_PATH}'. "
        "Run nn.py to train and save the model before using this module."
    )

_state = np.load(_MODEL_PATH)

W1    = _state["W1"].astype(np.float32)
W2    = _state["W2"].astype(np.float32)
x_min = _state["x_min"].astype(np.float32)
x_max = _state["x_max"].astype(np.float32)

print(f"[core_inference] Model loaded from {os.path.basename(_MODEL_PATH)}")
print(f"[core_inference] W1 shape: {W1.shape}  |  W2 shape: {W2.shape}")


# ============================================================================
# Inference Function
# ============================================================================

def run_inference(features_list):
    """
    Run a single forward pass through the C++ tensor engine.

    Parameters
    ----------
    features_list : list of float
        Exactly 9 numerical feature values in the same order used during
        training (see fps_data.csv column layout).

    Returns
    -------
    dict
        {
            "predicted_fps": float,
            "inference_time_ms": float,
        }

    Raises
    ------
    ValueError
        If `features_list` does not contain exactly 9 elements.
    """

    if len(features_list) != 9:
        raise ValueError(
            f"Expected exactly 9 features, received {len(features_list)}."
        )

    # Convert to 2D numpy array (batch size 1).
    X = np.array(features_list, dtype=np.float32).reshape(1, 9)

    # Normalize using the same min/max statistics saved during training.
    X_norm = ((X - x_min) / (x_max - x_min + 1e-8)).astype(np.float32)

    # ------------------------------------------------------------------
    # C++ Engine forward pass via custom Tensor backend
    # ------------------------------------------------------------------
    pool = MemoryPool(10 * 1024 * 1024)  # 10 MB capacity

    t_start = time.perf_counter()

    X_t  = Tensor.from_numpy(pool, X_norm)
    W1_t = Tensor.from_numpy(pool, W1)
    W2_t = Tensor.from_numpy(pool, W2)

    hidden = X_t.mul_naive(W1_t).relu()
    pred   = hidden.mul_naive(W2_t)

    fps_value = float(pred.numpy()[0][0])

    t_end = time.perf_counter()
    inference_time_ms = round((t_end - t_start) * 1000.0, 4)

    # Release pooled memory.
    pool.zero()

    return {
        "predicted_fps": fps_value,
        "inference_time_ms": inference_time_ms,
    }


# ============================================================================
# Standalone Testing
# ============================================================================

if __name__ == "__main__":
    print("--- core_inference.py standalone test ---\n")

    sample_features = [8, 3.5, 16, 1500, 32, 1.0, 2, 2, 2]
    print(f"Input features: {sample_features}")

    result = run_inference(sample_features)

    print(f"Predicted FPS : {result['predicted_fps']:.4f}")
    print(f"Inference time: {result['inference_time_ms']:.4f} ms")

    print("\n--- test complete ---")
