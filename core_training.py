"""
core_training.py
----------------
Decoupled backend logic for the Streamlit "Live Brain" training visualizer.

Executes the REAL training loop using the custom C++ tensor engine with
actual dataset loading, MemoryPool allocation, forward/backward passes
via the ComputationGraph, and gradient descent steps.

Zero mocked data. Zero simulated delays. All metrics are derived from
actual engine execution.
"""

import os
import sys
import time
import numpy as np

# ---------------------------------------------------------------------------
# Resolve the files/ directory so we can import the C++ tensor backend
# and locate the dataset.
# ---------------------------------------------------------------------------
_FILES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "files")
if _FILES_DIR not in sys.path:
    sys.path.insert(0, _FILES_DIR)

from tensor import MemoryPool, Tensor, ComputationGraph
from wrapper import SCBackend, sc_tensor_mse_loss


# ============================================================================
# Constants (mirrors nn.py configuration)
# ============================================================================

CSV_PATH       = os.path.join(_FILES_DIR, "fps_data.csv")
MODEL_PATH     = os.path.join(_FILES_DIR, "model_state.npz")

HIDDEN_DIM     = 16
LR             = 0.05

POOL_SIZE      = 50 * 1024 * 1024   # 50 MB main pool
META_POOL_SIZE = 10 * 1024 * 1024   # 10 MB metadata pool
GRAD_POOL_SIZE = 50 * 1024 * 1024   # 50 MB gradient pool


# ============================================================================
# Data Utilities (identical to nn.py)
# ============================================================================

def _load_csv(path):
    """Load fps_data.csv -> (X, Y) as float32 arrays."""
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Dataset not found at '{path}'.\n"
            "Run `python prepare_data.py` first to create fps_data.csv."
        )
    data = np.loadtxt(path, delimiter=",", skiprows=1, dtype=np.float32)
    return data[:, :-1], data[:, -1:]


def _compute_norm_params(arr, axis=0):
    return arr.min(axis=axis, keepdims=True), arr.max(axis=axis, keepdims=True)


def _normalize(arr, vmin, vmax):
    return (arr - vmin) / (vmax - vmin + 1e-8)


# ============================================================================
# Generator: Real Training Loop
# ============================================================================

def run_training_loop(epochs=500, learning_rate=None, save_model=True):
    """
    Generator that executes the real C++ engine training loop.

    Loads the dataset, initializes memory pools, builds the computation
    graph with the custom C++ backend, and runs actual gradient descent.
    Each epoch yields a dictionary of real performance metrics.

    Parameters
    ----------
    epochs : int, optional
        Number of training epochs. Default is 500 (matches nn.py).
    learning_rate : float or None, optional
        SGD learning rate. Defaults to module-level LR (0.05).
    save_model : bool, optional
        If True, save trained weights to model_state.npz after completion.

    Yields
    ------
    dict
        {
            "epoch"          : int   -- 1-indexed epoch number.
            "total_epochs"   : int   -- total epochs requested.
            "loss"           : float -- actual MSE loss from the C++ engine.
            "time_ms"        : float -- wall-clock time for this epoch (ms).
            "learning_phase" : int   -- phase indicator (0, 1, or 2).
            "num_samples"    : int   -- number of training samples.
            "num_features"   : int   -- number of input features.
        }
    """
    lr = learning_rate if learning_rate is not None else LR

    # -- Memory Pools --------------------------------------------------------
    pool_cpu  = MemoryPool(capacity_bytes=POOL_SIZE)
    meta_pool = MemoryPool(capacity_bytes=META_POOL_SIZE)
    grad_pool = MemoryPool(capacity_bytes=GRAD_POOL_SIZE)

    # -- Load Dataset --------------------------------------------------------
    X_raw, Y_raw = _load_csv(CSV_PATH)
    num_samples, num_features = X_raw.shape

    # -- Normalize -----------------------------------------------------------
    x_min, x_max = _compute_norm_params(X_raw, axis=0)
    y_min, y_max = _compute_norm_params(Y_raw, axis=0)

    X_np = _normalize(X_raw, x_min, x_max).astype(np.float32)
    Y_np = _normalize(Y_raw, y_min, y_max).astype(np.float32)

    # -- Create Tensors (C++ engine allocation) ------------------------------
    X        = Tensor.from_numpy(pool_cpu, X_np, requires_grad=True)
    Y_target = Tensor.from_numpy(pool_cpu, Y_np, requires_grad=True)

    W1 = Tensor.random_normal(pool_cpu, shape=[num_features, HIDDEN_DIM],
                              std=0.1, requires_grad=True)
    W2 = Tensor.random_normal(pool_cpu, shape=[HIDDEN_DIM, 1],
                              std=0.1, requires_grad=True)

    # -- Forward Pass (defines the computational graph) ----------------------
    X_act       = X.relu()
    hidden      = X_act.mul_naive(W1).relu()
    predictions = hidden.mul_naive(W2)

    loss_handle = sc_tensor_mse_loss(
        pool_cpu.handle, predictions.handle, Y_target.handle
    )
    loss_tensor = Tensor(pool_cpu, loss_handle, requires_grad=True)

    # -- Build Computation Graph ---------------------------------------------
    graph = ComputationGraph.build(
        meta_pool=meta_pool,
        loss=loss_tensor,
        pool_gpu=None,
        pool_grad_cpu=grad_pool,
        pool_grad_gpu=None,
        backend=SCBackend.CPU,
    )

    # -- Training Loop (real gradient descent) -------------------------------
    for epoch in range(1, epochs + 1):

        t_start = time.perf_counter()
        graph.step(learning_rate=lr)
        t_end = time.perf_counter()

        loss = float(graph.loss)
        time_ms = round((t_end - t_start) * 1000.0, 4)

        # Determine learning phase from actual loss magnitude.
        if loss > 0.1:
            learning_phase = 0   # High loss -- network is in early exploration
        elif loss > 0.03:
            learning_phase = 1   # Converging -- patterns being learned
        else:
            learning_phase = 2   # Converged  -- weights near-optimal

        yield {
            "epoch":          epoch,
            "total_epochs":   epochs,
            "loss":           loss,
            "time_ms":        time_ms,
            "learning_phase": learning_phase,
            "num_samples":    num_samples,
            "num_features":   num_features,
        }

    # -- Save Model (after full training completes) --------------------------
    if save_model:
        final_loss = float(graph.loss)
        np.savez(
            MODEL_PATH,
            W1=W1.numpy(),
            W2=W2.numpy(),
            x_min=x_min,
            x_max=x_max,
            y_min=y_min,
            y_max=y_max,
            final_loss=np.float64(final_loss),
            num_samples=np.int64(num_samples),
        )


# ============================================================================
# Standalone Testing
# ============================================================================

if __name__ == "__main__":
    print("--- core_training.py standalone test (real C++ engine) ---")
    print(f"Dataset: {os.path.basename(CSV_PATH)}")
    print(f"Running 10 epochs with LR={LR}\n")

    for metrics in run_training_loop(epochs=10, save_model=False):
        print(
            f"Epoch {metrics['epoch']:>3d}/{metrics['total_epochs']}  |  "
            f"Loss: {metrics['loss']:.6f}  |  "
            f"Time: {metrics['time_ms']:>8.4f} ms  |  "
            f"Phase: {metrics['learning_phase']}  |  "
            f"Samples: {metrics['num_samples']}"
        )

    print("\n--- test complete ---")
