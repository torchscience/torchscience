Roadmap
=======

This document outlines the development roadmap for torchscience.

Vision
------

torchscience provides PyTorch operators for mathematics, science, and engineering
with **deep PyTorch integration** as the core differentiator. Unlike scipy (no autograd)
or ad-hoc implementations, torchscience operators work seamlessly with:

* ``autograd`` — First, second, and higher-order gradients
* ``torch.compile`` — Graph mode optimization
* ``autocast`` — Mixed precision (float16, bfloat16)
* ``torch.func.vmap`` — Automatic batching
* ``complex`` — Complex tensor support
* ``sparse`` — COO and CSR sparse tensors
* ``quantized`` — Quantized tensor support
* ``tensordict`` — Structured return values
* ``nested`` — Variable-length outputs
* ``meta`` — Shape inference for AOT compilation

Scope
-----

**In scope:**

* Ordinary and partial differential equations
* Finite element methods
* Signal processing
* Statistics and probability
* Graphics and rendering primitives
* Numerical optimization
* Special mathematical functions
* Computational geometry
* Wavelets and integral transforms

**Out of scope:**

* Higher-level physics simulation (use dedicated libraries)
* Image processing pipelines (use torchvision)
* Machine learning models (use torch.nn)

Operator Categories
-------------------

Each operator belongs to a category based on its input/output shape behavior:

.. list-table::
   :header-rows: 1
   :widths: 20 50 30

   * - Category
     - Definition
     - Example
   * - Pointwise
     - Element-wise with broadcasting
     - ``gamma``, ``binomial_coefficient``
   * - Reduction
     - Reduces one or more dimensions
     - ``kurtosis``, ``kullback_leibler_divergence``
   * - Factory
     - Creates tensors from parameters
     - ``rectangular_window``, ``pink_noise``
   * - Fixed
     - Operates on specific dimensions
     - ``hilbert_transform``, ``dormand_prince_5``
   * - Batched
     - Batch dims + fixed trailing dims
     - ``matrix_exponential``, ``cubic_spline_evaluate``
   * - Identity
     - Preserves shape exactly
     - ``srgb_to_hsv``
   * - Flatten
     - Treats input as 1D internally
     - ``histogram``
   * - Dynamic
     - Output shape depends on values
     - ``convex_hull``, ``floyd_warshall``
   * - N-dimensional
     - Complex shape rules
     - ``minkowski_distance``, ``kd_tree``

Current Status
--------------

**Established modules:**

.. list-table::
   :header-rows: 1
   :widths: 30 40 30

   * - Module
     - Key Operators
     - Status
   * - ``special_functions``
     - gamma, beta, digamma, hypergeometric_2_f_1
     - Strong
   * - ``integration.initial_value_problem``
     - euler, runge_kutta_4, dormand_prince_5, backward_euler, adjoint
     - Strong
   * - ``polynomial``
     - Polynomial class, arithmetic, roots, calculus
     - Complete
   * - ``spline``
     - CubicSpline, BSpline, fitting, evaluation
     - Complete
   * - ``statistics``
     - t-tests, histogram, kurtosis
     - Partial
   * - ``signal_processing``
     - hilbert_transform, pink_noise, butterworth filter
     - Partial
   * - ``graphics``
     - color (srgb/hsv), shading (phong, cook_torrance), projection
     - Partial
   * - ``space_partitioning``
     - kd_tree, k_nearest_neighbors, range_search
     - Partial
   * - ``optimization``
     - brent, levenberg_marquardt, augmented_lagrangian, sinkhorn
     - Partial
   * - ``distance``
     - minkowski_distance
     - Minimal
   * - ``graph_theory``
     - floyd_warshall
     - Minimal
   * - ``information_theory``
     - kullback_leibler_divergence, jensen_shannon_divergence
     - Minimal
   * - ``combinatorics``
     - binomial_coefficient
     - Minimal

**Category coverage:**

.. list-table::
   :header-rows: 1
   :widths: 25 25 50

   * - Category
     - Status
     - Gap
   * - Pointwise
     - Complete
     - —
   * - Reduction
     - Complete
     - —
   * - Factory
     - Complete
     - —
   * - Fixed
     - Complete
     - —
   * - Batched
     - Partial
     - ``matrix_exponential``
   * - Identity
     - Complete
     - —
   * - Flatten
     - Complete
     - —
   * - Dynamic
     - Partial
     - ``convex_hull``
   * - N-dimensional
     - Complete
     - —

Phase 1: Complete Category Coverage
-----------------------------------

Fill remaining category gaps to demonstrate all operator patterns.

1.1 geometry.convex_hull
^^^^^^^^^^^^^^^^^^^^^^^^

**Category:** Dynamic

**Purpose:** Demonstrate variable-length output handling with nested tensors.

**Scope:**

* Quickhull algorithm (n-dimensional)
* Input: ``(n, d)`` points
* Output: ``(h, d)`` hull vertices, facets, simplices
* Nested tensor output for batched inputs

**PyTorch integration:**

* Nested tensors for variable-length results
* Meta tensors with bounded shape inference
* Autograd through vertex selection

1.2 linear_algebra.matrix_exponential
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Category:** Batched

**Purpose:** Demonstrate batched matrix operations with complex gradients.

**Scope:**

* Pade approximation with scaling and squaring
* Input: ``(..., n, n)`` matrices
* Output: ``(..., n, n)`` matrix exponentials
* Frechet derivative for efficient gradients

**PyTorch integration:**

* Arbitrary batch dimensions
* Complex matrix support
* Second-order gradients via Frechet derivative

Phase 2: Critical Module Establishment
--------------------------------------

Establish presence in missing high-value modules.

2.1 probability.distribution
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Dual API design:**

Distribution classes::

    dist = torchscience.probability.Normal(loc=0, scale=1)
    dist.log_prob(x)
    dist.cdf(x)
    dist.icdf(p)
    dist.sample(shape)
    dist.entropy()

Functional API::

    torchscience.probability.normal_pdf(x, loc, scale)
    torchscience.probability.normal_cdf(x, loc, scale)
    torchscience.probability.normal_quantile(p, loc, scale)

**Initial distributions:**

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Distribution
     - Dependencies
   * - Normal
     - erf (to implement)
   * - Exponential
     - —
   * - Gamma
     - gamma, incomplete_gamma
   * - Beta
     - beta, incomplete_beta (exists)
   * - StudentT
     - beta, gamma
   * - Chi2
     - gamma
   * - F
     - beta
   * - Uniform
     - —

2.2 wavelets
^^^^^^^^^^^^

**Scope:** Analysis only (transforms, not denoising utilities).

* Discrete Wavelet Transform (1D, 2D, n-D)
* Inverse DWT
* Wavelet families: Haar, Daubechies (db1-db20), Symlets, Coiflets
* Stationary wavelet transform (undecimated)
* Wavelet packet decomposition

2.3 integration.quadrature
^^^^^^^^^^^^^^^^^^^^^^^^^^

* Gauss-Legendre (fixed order)
* Gauss-Kronrod (adaptive 1D)
* Gauss-Laguerre, Gauss-Hermite (infinite domains)
* Cubature (n-D integration via tensor product or sparse grids)

2.4 differentiation
^^^^^^^^^^^^^^^^^^^

* Finite difference stencils (forward, backward, central)
* Arbitrary order derivatives
* Richardson extrapolation
* Gradient, Jacobian, Hessian utilities (complementing autograd)

Phase 3: Domain Depth
---------------------

Deepen coverage in established domains.

3.1 integration.bvp
^^^^^^^^^^^^^^^^^^^

Boundary value problem solvers:

* Shooting method (leverage IVP solvers)
* Collocation with splines (leverage spline module)
* Linear BVP direct solvers

3.2 finite_elements
^^^^^^^^^^^^^^^^^^^

n-dimensional finite element foundations:

* Lagrange shape functions (arbitrary order, arbitrary dimension)
* Gaussian quadrature nodes/weights (n-D tensor product)
* Element matrices (stiffness, mass, damping)
* Assembly routines (sparse matrix construction)
* Isoparametric mapping

3.3 integration.pde
^^^^^^^^^^^^^^^^^^^

Partial differential equation methods:

* Finite difference stencils (n-D Laplacian, gradient, divergence)
* Method of lines (reduce PDE to ODE system, use IVP solvers)
* Spectral methods (FFT-based)

3.4 geometry (extended)
^^^^^^^^^^^^^^^^^^^^^^^

Full computational geometry:

* **Transforms:** rotation_matrix, quaternion operations, affine transforms
* **Intersections:** ray-triangle, ray-sphere, ray-box, segment-segment
* **Structures:** Voronoi diagrams, Delaunay triangulation
* **Predicates:** orientation, in-circle, in-sphere

Phase 4: Extend Existing Modules
--------------------------------

Deepen coverage of established modules.

4.1 special_functions
^^^^^^^^^^^^^^^^^^^^^

* Bessel functions (J, Y, I, K, spherical, modified)
* Error functions (erf, erfc, erfi, Dawson)
* Elliptic integrals (complete K, E; incomplete F, E, Pi)
* Airy functions (Ai, Bi and derivatives)
* Exponential integrals (Ei, E1, En)

4.2 signal_processing
^^^^^^^^^^^^^^^^^^^^^

* Window functions: Hann, Hamming, Blackman, Kaiser, Gaussian, DPSS
* Filters: Chebyshev I/II, elliptic, Bessel, FIR design
* Spectral analysis: periodogram, Welch, spectrogram
* Resampling: interpolation, decimation

4.3 statistics
^^^^^^^^^^^^^^

* Regression: OLS, WLS, robust regression
* Resampling: bootstrap, jackknife, permutation tests
* Density estimation: kernel density estimation
* Tests: chi-squared, F-test, nonparametric tests

4.4 optimization
^^^^^^^^^^^^^^^^

* Quasi-Newton: L-BFGS, BFGS, SR1
* Trust region methods
* Sequential quadratic programming (SQP)
* Conjugate gradient methods

Implementation Order
--------------------

.. list-table::
   :header-rows: 1
   :widths: 5 25 30 20 20

   * - #
     - Module
     - Operator(s)
     - Category
     - Phase
   * - 1
     - ``geometry``
     - ``convex_hull``
     - Dynamic
     - 1
   * - 2
     - ``linear_algebra``
     - ``matrix_exponential``
     - Batched
     - 1
   * - 3
     - ``probability``
     - 8+ distributions
     - Pointwise
     - 2
   * - 4
     - ``wavelets``
     - DWT, IDWT, families
     - Fixed
     - 2
   * - 5
     - ``integration.quadrature``
     - gauss_kronrod, cubature
     - Reduction
     - 2
   * - 6
     - ``differentiation``
     - finite_difference
     - Fixed
     - 2
   * - 7
     - ``integration.bvp``
     - shooting, collocation
     - Fixed
     - 3
   * - 8
     - ``finite_elements``
     - shape_functions, assemble
     - Factory/Batched
     - 3
   * - 9
     - ``integration.pde``
     - method_of_lines
     - Fixed
     - 3
   * - 10
     - ``geometry``
     - transforms, intersections
     - Various
     - 3
   * - 11+
     - Extensions
     - special_functions, signal, stats, optim
     - Various
     - 4

PyTorch Integration Requirements
--------------------------------

Each new operator must demonstrate integration with PyTorch features:

.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - Feature
     - Priority
     - Notes
   * - ``autograd``
     - Required
     - First and second-order gradients
   * - ``torch.compile``
     - Required
     - Graph mode compatibility
   * - ``autocast``
     - Required
     - Mixed precision support
   * - ``vmap``
     - Required
     - Automatic batching
   * - ``complex``
     - High
     - Where mathematically meaningful
   * - ``meta``
     - High
     - Shape inference for AOT compilation
   * - ``sparse``
     - Medium
     - Where applicable (e.g., FEM assembly)
   * - ``nested``
     - Medium
     - For dynamic-length outputs
   * - ``quantized``
     - Low
     - Specialized use cases
   * - ``named``
     - Low
     - Experimental in PyTorch
