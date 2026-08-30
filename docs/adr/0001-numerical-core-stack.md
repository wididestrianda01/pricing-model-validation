# Numerical core: NumPy/SciPy + Numba + JAX (Greeks only) + QuantLib benchmark

The core is Python-first, but "pure NumPy" and "JAX everywhere" both misread what a bank validation team actually uses. Research into a large bank's Front Office Model Validation team and current postings shows C++ is the production benchmark language, yet a focused project should not be built in C++/Rust. Decision: NumPy/SciPy/pandas as the base, Numba for the Monte Carlo/PDE inner loop, JAX scoped to automatic-differentiation pathwise Greeks (its one genuine edge), and QuantLib-Python as the independent challenger benchmark the from-scratch engines must agree with.

Considered Options:
- Pure NumPy/SciPy — simplest, but shows no performance awareness.
- JAX as the general engine — modern, but not what bank validation production uses; only its autodiff earns a place.
- C++/Rust implementation — closest to a real desk, but too heavy for a focused project; calling QuantLib gets the benchmark value without the build cost.
