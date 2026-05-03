# Deep Learning Backend Python Wrapper

Complete Python binding for your custom C++ Deep Learning backend using `ctypes`.

## Files

- **`wrapper.py`**: Low-level `ctypes` bindings (C function mappings)
- **`tensor.py`**: High-level PyTorch-like OOP API
- **`example_usage.py`**: Comprehensive examples demonstrating all features

## Critical Fixes Implemented

### 1. 64-bit Pointer Truncation (DEADLYSIGNAL)

**Problem**: Without proper `restype` declaration, ctypes defaults to `c_int` (32-bit), truncating 64-bit pointers and causing AddressSanitizer crashes.

**Solution**: Every function returning a pointer has explicit `restype = ctypes.c_void_p`:

```python
lib.tensor_create.restype = ctypes.c_void_p  # CRITICAL
lib.tensor_pool_create.restype = ctypes.c_void_p  # CRITICAL
lib.tensor_add.restype = ctypes.c_void_p  # CRITICAL
# ... and all other pointer-returning functions
```

### 2. C++ Default Arguments

**Problem**: `ctypes` doesn't handle C++ default arguments. The function signature:
```cpp
tensor_pool_t *tensor_pool_create(size_t capacity_bytes, bool isOfDevice = false)
```

**Solution**: Explicitly declare both arguments:
```python
lib.tensor_pool_create.argtypes = [ctypes.c_size_t, ctypes.c_bool]
```

And always pass both when calling:
```python
pool = lib.tensor_pool_create(ctypes.c_size_t(size), ctypes.c_bool(False))
```

### 3. Shape Unpacking Bug

**Problem**: Passing a Python list directly to C array constructor fails.

**Incorrect**:
```python
dims_array = (ctypes.c_uint32 * ndims)(shape)  # WRONG!
```

**Correct**:
```python
dims_array = (ctypes.c_uint32 * ndims)(*shape)  # Use asterisk to unpack
```

### 4. ffi_prep_cif_var Errors

**Problem**: Not passing the boolean argument causes FFI errors.

**Solution**: Always explicitly pass `ctypes.c_bool(False)` or `ctypes.c_bool(True)`:
```python
pool = lib.tensor_pool_create(
    ctypes.c_size_t(capacity_bytes),
    ctypes.c_bool(is_device)  # Don't omit this!
)
```

## Quick Start

```python
from tensor import MemoryPool, Tensor, tensor, zeros, ones, randn

# 1. Create a memory pool (100MB for CPU)
pool = MemoryPool(capacity_bytes=100 * 1024 * 1024, is_device=False)

# 2. Create tensors
a = tensor(pool, [[1.0, 2.0, 3.0],
                  [4.0, 5.0, 6.0]])

b = zeros(pool, [2, 3])  # 2x3 zero matrix
c = ones(pool, [2, 3])   # 2x3 ones matrix
d = randn(pool, [2, 3])  # Random normal values

# 3. Perform operations using operator overloading
result = a + b           # Addition
result = a * b           # Matrix multiplication
result = a @ b           # Alternative matmul syntax

# 4. Use methods
a_t = a.transpose()      # Transpose
a_relu = a.relu()        # ReLU activation

# 5. Access data
print(a.shape)           # (2, 3)
print(a.data)            # Python list
print(a.to_numpy())      # NumPy array

# 6. Memory management
print(f"Pool used: {pool.used}/{pool.capacity} bytes")
pool.zero()              # Reset pool
pool.destroy()           # Free memory
```

## API Reference

### MemoryPool Class

```python
pool = MemoryPool(capacity_bytes, is_device=False)
```

**Properties**:
- `pool.capacity` - Total capacity in bytes
- `pool.used` - Used bytes
- `pool.available` - Available bytes

**Methods**:
- `pool.zero()` - Reset pool (invalidate all tensors)
- `pool.destroy()` - Free memory to OS

### Tensor Class

```python
t = Tensor(pool, data=None, shape=None, dtype=TensorDType.FLOAT32_T)
```

**Properties**:
- `t.id` - Unique tensor ID
- `t.ndims` - Number of dimensions (rank)
- `t.shape` - Shape tuple
- `t.size` - Total number of elements
- `t.data` - Python list representation

**Operators**:
- `a + b` - Element-wise addition
- `a * b` - Matrix multiplication
- `a @ b` - Matrix multiplication (alternative)

**Methods**:
- `t.transpose()` - Transpose
- `t.relu()` - ReLU activation
- `t.add_bias(bias)` - Add bias with broadcasting
- `t.mse_loss(target)` - MSE loss
- `t.cross_entropy_loss(target)` - Cross-entropy loss
- `t.backward()` - Backward pass (gradient computation)
- `t.fill_random_normal(mean, std_dev)` - Fill with random values
- `t.to_numpy()` - Convert to NumPy array
- `t.print()` - Print using C++ function

### Convenience Functions

```python
zeros(pool, shape, dtype=TensorDType.FLOAT32_T)
ones(pool, shape, dtype=TensorDType.FLOAT32_T)
randn(pool, shape, mean=0.0, std_dev=1.0)
tensor(pool, data, dtype=TensorDType.FLOAT32_T)
```

## Data Types

```python
from wrapper import TensorDType

TensorDType.UINT32_T
TensorDType.INT32_T
TensorDType.UINT64_T
TensorDType.INT64_T
TensorDType.FLOAT32_T  # Default
TensorDType.FLOAT64_T
```

## Complete Example: Neural Network Forward Pass

```python
from tensor import MemoryPool, tensor, randn

# Create pools
pool = MemoryPool(capacity_bytes=100 * 1024 * 1024)

# Input data (batch_size=2, features=3)
x = tensor(pool, [[1.0, 2.0, 3.0],
                  [4.0, 5.0, 6.0]])

# Layer 1: Linear (3 -> 4)
w1 = randn(pool, [3, 4], mean=0.0, std_dev=0.1)
b1 = randn(pool, [4], mean=0.0, std_dev=0.01)

# Layer 2: Linear (4 -> 2)
w2 = randn(pool, [4, 2], mean=0.0, std_dev=0.1)
b2 = randn(pool, [2], mean=0.0, std_dev=0.01)

# Forward pass
h1 = (x @ w1).add_bias(b1).relu()  # Hidden layer with ReLU
output = (h1 @ w2).add_bias(b2)    # Output layer

print(f"Input shape: {x.shape}")
print(f"Hidden shape: {h1.shape}")
print(f"Output shape: {output.shape}")
print(f"Output:\n{output.data}")

# Compute loss
targets = tensor(pool, [[1.0, 0.0],
                        [0.0, 1.0]])
loss = output.mse_loss(targets)
print(f"Loss: {loss.data}")
```

## Important Notes

1. **Memory Management**: The pool uses a bump allocator. Calling `pool.zero()` invalidates all tensors created from that pool.

2. **Pointer Lifetime**: Tensors hold raw pointers. Ensure the pool outlives all tensors created from it.

3. **std::vector Functions**: Functions taking `std::vector` (like `verifyIfDAG`, `assignBackendGraph`) are NOT mapped in `wrapper.py` as they require C++ ABI compatibility. Use C-compatible functions only.

4. **GPU Operations**: Set `is_device=True` when creating a GPU pool:
   ```python
   gpu_pool = MemoryPool(capacity_bytes=1024*1024*1024, is_device=True)
   ```

5. **Thread Safety**: The C++ backend's thread safety depends on your implementation. The Python wrapper adds no synchronization.

## Troubleshooting

### DEADLYSIGNAL / AddressSanitizer Errors
- Ensure all pointer-returning functions have `restype = ctypes.c_void_p`
- Check that you're not mixing pools (create tensor in pool A, use in pool B)

### "ffi_prep_cif_var" Errors
- Always pass both arguments to `tensor_pool_create`
- Use `ctypes.c_bool(False)` explicitly

### Segmentation Faults
- Verify the shared library path is correct
- Check that tensor data matches the shape (total elements)
- Don't access tensors after calling `pool.zero()`

### Shape Errors
- Use asterisk unpacking: `(c_uint32 * n)(*shape)`
- Ensure shape dimensions don't exceed `TENSOR_MAX_DIMS` (8)

## Testing

Run the comprehensive example file:

```bash
python example_usage.py
```

This will execute 8 different examples covering:
- Basic tensor operations
- Operator overloading
- Matrix operations
- Bias addition and broadcasting
- Loss functions
- Memory pool management
- NumPy integration
- Chained operations

## License

This wrapper code matches the license of your C++ backend.
