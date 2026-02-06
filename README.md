# JAX Programming

JAX programming practices made by LLMs

## Table of Contents
- [What is JAX?](#what-is-jax)
- [Installation](#installation)
- [JAX Basics](#jax-basics)
- [Core Concepts](#core-concepts)
- [JAX Grammar & Transformations](#jax-grammar--transformations)
- [Common Patterns](#common-patterns)
- [Resources](#resources)

## What is JAX?

JAX is a Python library for high-performance numerical computing and machine learning research. It provides:

- **NumPy-like API**: Familiar syntax for array operations
- **Automatic Differentiation**: Compute gradients of Python functions
- **JIT Compilation**: Speed up code with XLA (Accelerated Linear Algebra)
- **Vectorization**: Automatically vectorize functions across batch dimensions
- **Parallelization**: Distribute computations across multiple devices

JAX is particularly popular for neural network research, scientific computing, and any application requiring fast numerical computation with automatic differentiation.

## Installation

```bash
# CPU-only version
pip install jax

# GPU version (CUDA 12)
pip install -U "jax[cuda12]"

# TPU version
pip install -U "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
```

## JAX Basics

### Importing JAX

```python
import jax
import jax.numpy as jnp
from jax import grad, jit, vmap, pmap
```

### Arrays in JAX

JAX arrays are similar to NumPy arrays but are immutable:

```python
# Create arrays
x = jnp.array([1, 2, 3, 4])
y = jnp.ones((3, 3))
z = jnp.zeros((2, 5))

# Array operations
a = jnp.dot(x, x)
b = jnp.sum(y, axis=0)
c = jnp.reshape(z, (10,))
```

**Important**: JAX arrays are immutable! Instead of in-place operations, use `.at[]` syntax:

```python
# Wrong: x[0] = 10  # This will fail!
# Correct:
x = x.at[0].set(10)        # Set index 0 to 10
x = x.at[1].add(5)         # Add 5 to index 1
x = x.at[0:2].multiply(2)  # Multiply indices 0-1 by 2
```

## Core Concepts

### 1. Pure Functions

JAX transformations require pure functions (no side effects):

```python
# Good: Pure function
def pure_function(x):
    return x ** 2 + 3 * x + 1

# Bad: Has side effects
global_list = []
def impure_function(x):
    global_list.append(x)  # Side effect!
    return x ** 2
```

### 2. Random Numbers

JAX uses explicit PRNG keys for reproducibility:

```python
from jax import random

# Create a random key
key = random.PRNGKey(0)

# Generate random numbers
key, subkey = random.split(key)
random_array = random.normal(subkey, shape=(3, 3))

# Split key for multiple random operations
keys = random.split(key, num=3)
```

### 3. Data Types

JAX defaults to 32-bit precision for performance:

```python
# Default is float32
x = jnp.array([1.0, 2.0, 3.0])
print(x.dtype)  # dtype('float32')

# Explicit type casting
x_64 = x.astype(jnp.float64)
x_int = x.astype(jnp.int32)
```

## JAX Grammar & Transformations

### 1. `jit` - Just-In-Time Compilation

Compile functions for faster execution:

```python
from jax import jit

# Regular function
def slow_function(x):
    return jnp.sum(x ** 2)

# JIT-compiled function
@jit
def fast_function(x):
    return jnp.sum(x ** 2)

# Or use directly
fast_function = jit(slow_function)

# Usage
x = jnp.arange(1000)
result = fast_function(x)  # First call: compiles, then runs
result = fast_function(x)  # Subsequent calls: just runs (fast!)
```

### 2. `grad` - Automatic Differentiation

Compute gradients automatically:

```python
from jax import grad

# Define a function
def f(x):
    return x ** 3 + 2 * x ** 2 + 3 * x + 4

# Get gradient function (derivative)
df_dx = grad(f)

# Compute gradient at x=1.0
gradient = df_dx(1.0)  # Returns: 3*(1)^2 + 4*(1) + 3 = 10.0

# Gradients of multivariable functions
def g(x, y):
    return x ** 2 * y + y ** 3

# Gradient w.r.t. first argument (x)
dg_dx = grad(g, argnums=0)

# Gradient w.r.t. both arguments
dg_dxy = grad(g, argnums=(0, 1))
```

### 3. `value_and_grad` - Get Both Value and Gradient

```python
from jax import value_and_grad

def loss_fn(params, x, y):
    prediction = params['w'] * x + params['b']
    return jnp.mean((prediction - y) ** 2)

# Get both loss value and gradient
loss_and_grad_fn = value_and_grad(loss_fn)
loss, grads = loss_and_grad_fn(params, x_data, y_data)
```

### 4. `vmap` - Automatic Vectorization

Vectorize functions across batch dimensions:

```python
from jax import vmap

# Function that works on single example
def predict(params, x):
    return jnp.dot(params['w'], x) + params['b']

# Vectorize across batch dimension
predict_batch = vmap(predict, in_axes=(None, 0))

# Usage
params = {'w': jnp.array([1., 2., 3.]), 'b': 0.5}
x_batch = jnp.array([[1., 2., 3.],
                      [4., 5., 6.],
                      [7., 8., 9.]])
predictions = predict_batch(params, x_batch)

# vmap can handle multiple batch dimensions
def pairwise_distance(x, y):
    return jnp.linalg.norm(x - y)

# Vectorize over both inputs
pairwise_distances = vmap(vmap(pairwise_distance, (None, 0)), (0, None))
```

### 5. `pmap` - Parallel Map

Distribute computation across multiple devices (GPUs/TPUs):

```python
from jax import pmap

# Function to parallelize
def compute(x):
    return jnp.sum(x ** 2)

# Parallelize across devices
parallel_compute = pmap(compute)

# Data must be sharded across devices
n_devices = jax.device_count()
x = jnp.arange(n_devices * 100).reshape((n_devices, 100))
results = parallel_compute(x)
```

### 6. `jvp` and `vjp` - Jacobian Products

```python
from jax import jvp, vjp

# Forward-mode autodiff (JVP)
def f(x):
    return x ** 3

x = 2.0
v = 1.0  # tangent vector
y, y_dot = jvp(f, (x,), (v,))  # y = f(x), y_dot = df/dx * v

# Reverse-mode autodiff (VJP)
y, f_vjp = vjp(f, x)
x_bar, = f_vjp(1.0)  # Compute gradient
```

### 7. `scan` - Efficient Loops

For loops that carry state:

```python
from jax.lax import scan

# Cumulative sum using scan
def cumsum(carry, x):
    new_carry = carry + x
    return new_carry, new_carry

init_carry = 0
xs = jnp.array([1, 2, 3, 4, 5])
final_carry, ys = scan(cumsum, init_carry, xs)
# ys = [1, 3, 6, 10, 15]
```

### 8. `cond` - Conditional Operations

Conditionals in JIT-compiled code:

```python
from jax.lax import cond

def true_fn(x):
    return x + 1

def false_fn(x):
    return x - 1

result = cond(x > 0, true_fn, false_fn, x)
```

## Common Patterns

### 1. Neural Network Layer

```python
def dense_layer(params, x):
    """Simple dense layer: y = W @ x + b"""
    return jnp.dot(x, params['w']) + params['b']

# Vectorize for batch
dense_batch = vmap(dense_layer, in_axes=(None, 0))

# JIT compile for speed
dense_batch = jit(dense_batch)
```

### 2. Training Loop

```python
@jit
def update(params, x, y, learning_rate=0.01):
    """Single training step"""
    loss, grads = value_and_grad(loss_fn)(params, x, y)
    # Update parameters
    params = jax.tree.map(lambda p, g: p - learning_rate * g, params, grads)
    return params, loss

# Training loop
for epoch in range(num_epochs):
    for batch_x, batch_y in dataloader:
        params, loss = update(params, batch_x, batch_y)
        print(f"Loss: {loss}")
```

### 3. Working with PyTrees

JAX can work with nested structures (PyTrees):

```python
import jax.tree_util as tree

# Example PyTree (nested dict)
params = {
    'layer1': {'w': jnp.ones((10, 5)), 'b': jnp.zeros(5)},
    'layer2': {'w': jnp.ones((5, 1)), 'b': jnp.zeros(1)}
}

# Map function over all leaves
def init_zero(x):
    return jnp.zeros_like(x)

zero_params = tree.tree_map(init_zero, params)

# Add two PyTrees
params_sum = tree.tree_map(lambda x, y: x + y, params, zero_params)
```

### 4. Checkpointing Gradients

For memory efficiency in deep networks:

```python
from jax.checkpoint import checkpoint

@checkpoint
def expensive_layer(x):
    # Complex computation
    return jnp.tanh(jnp.dot(x, x.T))

# This will recompute forward pass during backward pass
# instead of storing all intermediate activations
```

### 5. Custom Gradients

Define custom gradient rules:

```python
from jax import custom_vjp

@custom_vjp
def f(x):
    return jnp.sin(x)

def f_fwd(x):
    return f(x), x

def f_bwd(x, g):
    return (g * jnp.cos(x),)

f.defvjp(f_fwd, f_bwd)
```

## Resources

### Official Documentation
- [JAX Documentation](https://jax.readthedocs.io/)
- [JAX GitHub Repository](https://github.com/google/jax)
- [JAX Tutorials](https://jax.readthedocs.io/en/latest/notebooks/quickstart.html)

### Learning Resources
- [JAX 101 Tutorial](https://jax.readthedocs.io/en/latest/jax-101/index.html)
- [Common Gotchas](https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html)
- [Thinking in JAX](https://jax.readthedocs.io/en/latest/notebooks/thinking_in_jax.html)

### Community
- [JAX Discussions](https://github.com/google/jax/discussions)
- [JAX Issues](https://github.com/google/jax/issues)

---

**Note**: This repository contains JAX programming practices and examples developed with assistance from Large Language Models (LLMs).
