# JAX Programming

A comprehensive guide to JAX programming practices, focusing on core concepts, programming grammars, and best practices for high-performance numerical computing.

## Table of Contents
- [Introduction](#introduction)
- [Installation](#installation)
- [JAX Basics](#jax-basics)
- [Programming Grammar](#programming-grammar)
- [Common Patterns](#common-patterns)
- [Best Practices](#best-practices)
- [Resources](#resources)

## Introduction

JAX is a high-performance numerical computing library developed by Google Research. It brings together:
- **NumPy-like API**: Familiar interface for scientific computing
- **Automatic Differentiation**: Compute gradients of Python and NumPy code
- **JIT Compilation**: Accelerate code execution using XLA (Accelerated Linear Algebra)
- **Vectorization**: Automatically vectorize functions for batch processing
- **Hardware Acceleration**: Run code on CPUs, GPUs, and TPUs

This repository contains JAX programming practices, examples, and patterns to help you master JAX.

## Installation

### Basic Installation
```bash
# Install JAX with CPU support
pip install jax jaxlib

# Install JAX with GPU support (CUDA 12)
pip install -U "jax[cuda12]"

# Install JAX with TPU support
pip install -U "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
```

### Verify Installation
```python
import jax
import jax.numpy as jnp

print(f"JAX version: {jax.__version__}")
print(f"Available devices: {jax.devices()}")
```

## JAX Basics

### 1. JAX Arrays (jax.numpy)

JAX provides a NumPy-compatible API through `jax.numpy`:

```python
import jax.numpy as jnp

# Creating arrays
x = jnp.array([1, 2, 3, 4, 5])
y = jnp.linspace(0, 1, 100)
z = jnp.zeros((3, 4))

# Array operations (similar to NumPy)
result = jnp.dot(x, x)
mean = jnp.mean(x)
reshaped = x.reshape(5, 1)
```

**Key Difference from NumPy**: JAX arrays are **immutable**. Operations return new arrays instead of modifying in place.

```python
# This will NOT work in JAX (unlike NumPy)
# x[0] = 10  # Error: JAX arrays are immutable

# Instead, use .at[] syntax for updates
x = x.at[0].set(10)
```

### 2. Automatic Differentiation

JAX provides powerful automatic differentiation through `grad`, `value_and_grad`, and related functions:

```python
from jax import grad, value_and_grad
import jax.numpy as jnp

# Define a function
def f(x):
    return x**3 + 2*x**2 - 5*x + 3

# Compute gradient
df_dx = grad(f)
gradient_at_2 = df_dx(2.0)  # d/dx at x=2

# Get both value and gradient
value, gradient = value_and_grad(f)(2.0)

# Multiple arguments
def multi_arg_func(x, y):
    return x**2 + y**2

# Gradient with respect to first argument
grad_x = grad(multi_arg_func, argnums=0)
# Gradient with respect to both arguments
grad_both = grad(multi_arg_func, argnums=(0, 1))
```

### 3. Just-In-Time (JIT) Compilation

Use `@jax.jit` to compile functions for faster execution:

```python
import jax
import jax.numpy as jnp

# Regular function
def slow_function(x):
    return jnp.sum(x**2)

# JIT-compiled function
@jax.jit
def fast_function(x):
    return jnp.sum(x**2)

# Usage
x = jnp.arange(1000000)
result = fast_function(x)  # Compiled on first call, fast on subsequent calls
```

### 4. Vectorization (vmap)

Automatically vectorize functions to operate over batches:

```python
from jax import vmap
import jax.numpy as jnp

# Function that operates on single input
def process_single(x):
    return jnp.sum(x**2)

# Vectorized version that operates on batches
process_batch = vmap(process_single)

# Usage
batch = jnp.array([[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9]])
results = process_batch(batch)  # Applies to each row
```

### 5. Random Number Generation

JAX uses explicit PRNG keys for reproducible randomness:

```python
from jax import random
import jax.numpy as jnp

# Create a random key
key = random.PRNGKey(0)

# Split key for multiple random operations
key, subkey = random.split(key)

# Generate random numbers
random_array = random.normal(subkey, shape=(10,))
random_uniform = random.uniform(key, shape=(5, 5))

# Always split keys for new random operations
key, *subkeys = random.split(key, 3)
rand1 = random.normal(subkeys[0], shape=(100,))
rand2 = random.normal(subkeys[1], shape=(100,))
```

## Programming Grammar

### Function Transformations

JAX provides composable function transformations that follow a consistent grammar:

```python
import jax
import jax.numpy as jnp

# Basic transformations
jax.jit        # Just-in-time compilation
jax.grad       # Gradient computation
jax.vmap       # Vectorization
jax.pmap       # Parallelization across devices

# These can be composed
@jax.jit
@jax.vmap
@jax.grad
def combined_transform(x):
    return jnp.sum(x**2)
```

### Pure Functions

JAX transformations require **pure functions** - functions with no side effects:

```python
# ✅ Good: Pure function
def good_function(x):
    return x**2 + 2*x + 1

# ❌ Bad: Impure function (modifies external state)
counter = 0
def bad_function(x):
    global counter
    counter += 1  # Side effect!
    return x**2

# ❌ Bad: Non-deterministic
import random
def bad_random(x):
    return x + random.random()  # Use jax.random instead!
```

### Control Flow

Use JAX's functional control flow primitives instead of Python's:

```python
from jax import lax
import jax.numpy as jnp

# Conditional: lax.cond
def conditional_abs(x):
    return lax.cond(
        x >= 0,
        lambda x: x,      # true branch
        lambda x: -x,     # false branch
        x
    )

# Looping: lax.fori_loop
def sum_with_loop(n):
    def body(i, val):
        return val + i
    return lax.fori_loop(0, n, body, 0)

# While loop: lax.while_loop
def while_example(x):
    def cond_fun(val):
        return val < 10
    def body_fun(val):
        return val + 1
    return lax.while_loop(cond_fun, body_fun, x)

# Switch statement: lax.switch
def switch_example(index, x):
    branches = [
        lambda x: x**2,
        lambda x: x**3,
        lambda x: jnp.sqrt(x)
    ]
    return lax.switch(index, branches, x)
```

### PyTree Structures

JAX operates on PyTrees - nested structures of arrays:

```python
import jax
import jax.numpy as jnp

# PyTrees can be lists, tuples, dicts, or custom classes
pytree = {
    'weights': jnp.array([1.0, 2.0, 3.0]),
    'bias': jnp.array([0.5]),
    'layers': [
        jnp.array([[1, 2], [3, 4]]),
        jnp.array([[5, 6], [7, 8]])
    ]
}

# Many JAX functions work on PyTrees
def loss_fn(params):
    return jnp.sum(params['weights']**2)

gradient = jax.grad(loss_fn)(pytree)

# tree_map applies a function to all leaves
pytree_doubled = jax.tree_map(lambda x: 2*x, pytree)
```

## Common Patterns

### 1. Neural Network Layer

```python
import jax
import jax.numpy as jnp
from jax import random

def init_layer(key, input_dim, output_dim):
    """Initialize a dense layer."""
    w_key, b_key = random.split(key)
    w = random.normal(w_key, (input_dim, output_dim)) * 0.01
    b = jnp.zeros(output_dim)
    return {'w': w, 'b': b}

@jax.jit
def forward(params, x):
    """Forward pass."""
    return jnp.dot(x, params['w']) + params['b']
```

### 2. Training Loop

```python
from jax import grad, jit
import jax.numpy as jnp

def loss_fn(params, x, y):
    """Compute loss."""
    pred = forward(params, x)
    return jnp.mean((pred - y)**2)

@jit
def update(params, x, y, lr=0.01):
    """Single training step."""
    grads = grad(loss_fn)(params, x, y)
    # Update parameters
    return jax.tree_map(lambda p, g: p - lr * g, params, grads)

# Training loop
for epoch in range(num_epochs):
    params = update(params, x_batch, y_batch)
```

### 3. Gradient Accumulation

```python
from jax import value_and_grad
import jax.numpy as jnp

@jax.jit
def compute_loss_and_grad(params, batch):
    loss, grads = value_and_grad(loss_fn)(params, batch['x'], batch['y'])
    return loss, grads

def train_step_with_accumulation(params, batches):
    """Accumulate gradients over multiple batches."""
    total_loss = 0.0
    accum_grads = jax.tree_map(jnp.zeros_like, params)
    
    for batch in batches:
        loss, grads = compute_loss_and_grad(params, batch)
        total_loss += loss
        accum_grads = jax.tree_map(lambda a, g: a + g, accum_grads, grads)
    
    # Average gradients
    n_batches = len(batches)
    avg_grads = jax.tree_map(lambda g: g / n_batches, accum_grads)
    
    return total_loss / n_batches, avg_grads
```

## Best Practices

### 1. Use JAX Transformations Correctly

- **Always use `jax.random` instead of `numpy.random` or Python's `random`**
- **Avoid in-place operations** - JAX arrays are immutable
- **Write pure functions** for transformations like `jit`, `grad`, and `vmap`

### 2. Performance Optimization

```python
# ✅ Good: JIT compile hot paths
@jax.jit
def fast_computation(x):
    return jnp.sum(x**2)

# ✅ Good: Use vmap for batching
process_batch = jax.vmap(process_single)

# ❌ Avoid: Python loops over arrays
def slow_sum(arr):
    total = 0
    for x in arr:  # Slow!
        total += x
    return total

# ✅ Better: Use JAX operations
def fast_sum(arr):
    return jnp.sum(arr)
```

### 3. Debugging

```python
# Use jax.debug.print for debugging JIT-compiled code
from jax import debug

@jax.jit
def debug_function(x):
    debug.print("x = {}", x)
    return x**2

# Disable JIT for debugging
with jax.disable_jit():
    result = my_jitted_function(x)

# Check for NaN values
jax.config.update("jax_debug_nans", True)
```

### 4. Memory Management

```python
# Clear compilation cache if memory is an issue
jax.clear_caches()

# Use float32 by default for better performance
jax.config.update("jax_default_dtype_bits", "32")

# Enable 64-bit precision when needed
jax.config.update("jax_enable_x64", True)
```

## Resources

### Official Documentation
- [JAX Documentation](https://jax.readthedocs.io/)
- [JAX GitHub Repository](https://github.com/google/jax)
- [JAX Quickstart](https://jax.readthedocs.io/en/latest/quickstart.html)

### Tutorials and Guides
- [JAX Tutorial by Google](https://jax.readthedocs.io/en/latest/tutorials.html)
- [The Autodiff Cookbook](https://jax.readthedocs.io/en/latest/notebooks/autodiff_cookbook.html)
- [Thinking in JAX](https://jax.readthedocs.io/en/latest/notebooks/thinking_in_jax.html)

### Libraries Built on JAX
- **Flax**: Neural network library
- **Optax**: Gradient processing and optimization
- **Haiku**: Neural network library by DeepMind
- **JAXopt**: Optimization library
- **Equinox**: Neural networks and scientific computing

### Community
- [JAX Discussions](https://github.com/google/jax/discussions)
- [JAX Issues](https://github.com/google/jax/issues)

## Contributing

This repository is a collection of JAX programming practices. Contributions are welcome!

## License

This project is open source and available for educational purposes.
