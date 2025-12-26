Roadmap
=======

This document outlines the development roadmap for torchscience.

MVP Roadmap
-----------

The MVP focuses on implementing the minimum operators necessary to establish:

1. **Module structure** — One or more operators in each planned top-level module
2. **Implementation patterns** — Examples of each operator category (pointwise, reduction, factory, fixed, n-dimensional, identity, flatten, dynamic, batched)

MVP Milestones
^^^^^^^^^^^^^^

**Milestone 1: Core Categories** ✓

* Pointwise operators (1-4 ary) via special functions
* Factory operators via window functions
* Fixed operators via signal processing

**Milestone 2: Remaining Categories**

* Reduction operators via ``kurtosis`` ✓
* Fixed operators via ``hilbert_transform`` ✓
* N-dimensional operators via ``minkowski_distance``
* Dynamic operators via ``shortest_path``
* Batched operators via ``matrix_exponential``
* Identity operators via ``srgb_to_hsv``
* Flatten operators via ``histogram`` ✓

**Milestone 3: Module Coverage**

* One operator in each planned module

Modules to Establish
^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 40 30 30

   * - Module
     - MVP Operator
     - Category
   * - ``clustering``
     - ``k_means``
     - N-dimensional
   * - ``combinatorics``
     - ``binomial_coefficient``
     - Pointwise
   * - ``differentiation``
     - ``finite_difference``
     - Fixed
   * - ``distance``
     - ``minkowski_distance``
     - N-dimensional
   * - ``finite_elements``
     - ``stiffness_matrix``
     - Factory
   * - ``geometry``
     - ``convex_hull``
     - Dynamic
   * - ``geometry.intersections``
     - ``ray_triangle_intersection``
     - Batched
   * - ``geometry.transforms``
     - ``rotation_matrix``
     - Factory
   * - ``graph_theory``
     - ``shortest_path``
     - Dynamic
   * - ``graphics.color``
     - ``srgb_to_hsv``
     - Identity
   * - ``graphics.shading``
     - ``cook_torrance``
     - Reduction
   * - ``information_theory``
     - ``kullback_leibler_divergence``
     - Reduction
   * - ``integral_transforms``
     - ``discrete_cosine_transform``
     - Fixed
   * - ``integration.ordinary_differential_equations``
     - ``runge_kutta_step``
     - Fixed
   * - ``integration.quadrature``
     - ``adaptive_quadrature``
     - Reduction
   * - ``interpolation``
     - ``cubic_spline_interpolate``
     - Fixed
   * - ``linear_algebra``
     - ``matrix_exponential``
     - Batched
   * - ``noise``
     - ``pink_noise``
     - Factory
   * - ``number_theory``
     - ``prime_sieve``
     - Dynamic
   * - ``optimization``
     - ``brent_minimization``
     - Reduction
   * - ``optimization.root_finding``
     - ``brent_root``
     - Reduction
   * - ``optimization.test_functions`` ✓
     - ``rosenbrock``
     - Reduction
   * - ``polynomials``
     - ``evaluate_polynomial``
     - Pointwise
   * - ``probability``
     - ``gaussian_probability_density``
     - Pointwise
   * - ``signal_processing.integral_transform`` ✓
     - ``hilbert_transform``
     - Fixed
   * - ``space_partitioning``
     - ``kd_tree_query``
     - N-dimensional
   * - ``statistics.descriptive`` ✓
     - ``kurtosis``, ``histogram``
     - Reduction, Flatten
   * - ``statistics.hypothesis_testing``
     - ``t_test``
     - Reduction
   * - ``wavelets``
     - ``discrete_wavelet_transform``
     - Fixed

Shape Behavior Reference
^^^^^^^^^^^^^^^^^^^^^^^^

Each operator's category is determined by its input/output shape behavior:

.. list-table::
   :header-rows: 1
   :widths: 25 25 20 20 10

   * - Module
     - Operator
     - Input Shape
     - Output Shape
     - Category
   * - ``clustering``
     - ``k_means``
     - ``(n, d)`` points
     - ``(n,)`` labels, ``(k, d)`` centroids
     - N-dimensional
   * - ``combinatorics``
     - ``binomial_coefficient``
     - ``(...)`` n, ``(...)`` k
     - ``(...)`` broadcasted
     - Pointwise
   * - ``differentiation``
     - ``finite_difference``
     - ``(..., n)``
     - ``(..., n)`` or ``(..., n-1)``
     - Fixed
   * - ``distance``
     - ``minkowski_distance``
     - ``(m, d)``, ``(n, d)``
     - ``(m, n)`` pairwise
     - N-dimensional
   * - ``finite_elements``
     - ``stiffness_matrix``
     - mesh/element info
     - ``(n_dof, n_dof)``
     - Factory
   * - ``geometry``
     - ``convex_hull``
     - ``(n, d)`` points
     - ``(h, d)`` hull vertices
     - Dynamic
   * - ``geometry.intersections``
     - ``ray_triangle_intersection``
     - ``(..., 3)`` rays, tris
     - ``(...)`` t, hit
     - Batched
   * - ``geometry.transforms``
     - ``rotation_matrix``
     - angle(s), axis
     - ``(..., d, d)``
     - Factory
   * - ``graph_theory``
     - ``shortest_path``
     - ``(n, n)`` adjacency
     - variable-length path
     - Dynamic
   * - ``graphics.color``
     - ``srgb_to_hsv``
     - ``(..., 3)``
     - ``(..., 3)``
     - Identity
   * - ``graphics.shading``
     - ``cook_torrance``
     - ``(..., 3)`` vectors
     - ``(...)`` reflectance
     - Reduction
   * - ``information_theory``
     - ``kullback_leibler_divergence``
     - ``(..., n)`` P, Q
     - ``(...)``
     - Reduction
   * - ``integral_transforms``
     - ``discrete_cosine_transform``
     - ``(..., n)``
     - ``(..., n)``
     - Fixed
   * - ``integration.ode``
     - ``runge_kutta_step``
     - ``(..., n)`` state
     - ``(..., n)`` next state
     - Fixed
   * - ``integration.quadrature``
     - ``adaptive_quadrature``
     - function, bounds
     - ``(...)`` integral
     - Reduction
   * - ``interpolation``
     - ``cubic_spline_interpolate``
     - ``(n,)`` x/y, ``(m,)`` query
     - ``(m,)`` interpolated
     - Fixed
   * - ``linear_algebra``
     - ``matrix_exponential``
     - ``(..., n, n)``
     - ``(..., n, n)``
     - Batched
   * - ``noise``
     - ``pink_noise``
     - shape specification
     - ``(...)`` noise
     - Factory
   * - ``number_theory``
     - ``prime_sieve``
     - integer n
     - ``(π(n),)`` primes
     - Dynamic
   * - ``optimization``
     - ``brent_minimization``
     - function, bounds
     - ``(...)`` x_min
     - Reduction
   * - ``optimization.root_finding``
     - ``brent_root``
     - function, bounds
     - ``(...)`` root
     - Reduction
   * - ``optimization.test_functions``
     - ``rosenbrock``
     - ``(..., n)``
     - ``(...)``
     - Reduction
   * - ``polynomials``
     - ``evaluate_polynomial``
     - ``(...)`` x, ``(k,)`` coeffs
     - ``(...)``
     - Pointwise
   * - ``probability``
     - ``gaussian_probability_density``
     - ``(...)`` x, μ, σ
     - ``(...)``
     - Pointwise
   * - ``signal_processing``
     - ``hilbert_transform``
     - ``(..., n)``
     - ``(..., n)`` complex
     - Fixed
   * - ``space_partitioning``
     - ``kd_tree_query``
     - ``(n, d)`` tree, ``(m, d)`` queries
     - ``(m, k)`` indices
     - N-dimensional
   * - ``statistics.descriptive``
     - ``kurtosis``
     - ``(..., n)``
     - ``(...)``
     - Reduction
   * - ``statistics.descriptive``
     - ``histogram``
     - ``(...)`` samples
     - ``(bins,)`` counts
     - Flatten
   * - ``statistics.hypothesis_testing``
     - ``t_test``
     - ``(..., n)`` samples
     - ``(...)`` statistic
     - Reduction
   * - ``wavelets``
     - ``discrete_wavelet_transform``
     - ``(..., n)``
     - ``(..., n)`` coeffs
     - Fixed

**Category Definitions:**

* **Pointwise** — Element-wise with broadcasting; each output element depends only on corresponding input element(s)
* **Reduction** — Reduces one or more dimensions; output has fewer dimensions than input
* **Factory** — Creates tensors without tensor inputs; only takes shape/parameter specifications
* **Fixed** — Operates on specific dimensions (e.g., last dim, last 2 dims); other dims are batch dims
* **Batched** — Arbitrary batch dimensions at start, fixed operation on trailing dimensions
* **Identity** — Preserves shape exactly; elements may interact but dimensions unchanged
* **Flatten** — Treats input as 1D internally regardless of actual shape
* **Dynamic** — Output shape depends on input data values, not just input shape
* **N-dimensional** — Works generically on arbitrary dimensionality with complex shape rules

----

Full Roadmap
------------

The following areas are planned for implementation. Each will follow the architectural patterns established in the current codebase.

Special Functions
^^^^^^^^^^^^^^^^^

Expanding coverage of mathematical special functions:

* Gamma and related functions (digamma, polygamma, beta, log-gamma, reciprocal gamma)
* Bessel functions (J, Y, I, K, spherical, modified)
* Error functions (erf, erfc, erfi, Dawson, Fresnel integrals)
* Elliptic integrals and functions (Jacobi, Weierstrass, complete/incomplete)
* Exponential integrals (Ei, E1, En, logarithmic integral, sine/cosine integrals)
* Orthogonal polynomials (Legendre, Laguerre, Hermite, Jacobi, Gegenbauer)
* Zeta and L-functions (Riemann zeta, Hurwitz zeta, polylogarithm, Dirichlet eta)
* Hypergeometric functions (confluent, generalized, Meijer G, Fox H)
* Airy and Scorer functions
* Parabolic cylinder functions
* Mathieu functions
* Spheroidal wave functions
* Coulomb wave functions
* Struve functions
* Kelvin functions
* Lommel functions

Signal Processing
^^^^^^^^^^^^^^^^^

Comprehensive signal processing toolkit:

* **Filters**: Butterworth, Chebyshev I/II, elliptic, Bessel, FIR design methods
* **Window functions**: Hann, Hamming, Blackman, Kaiser, Gaussian, Tukey, DPSS
* **Waveforms**: Square, sawtooth, triangle, chirp, pulse
* **Spectral analysis**: Periodogram, Welch, multitaper, spectrogram
* **Transforms**: Hilbert ✓, cepstrum, envelope detection
* **Resampling**: Interpolation, decimation, rational resampling
* **Convolution**: 1D/2D/3D convolution, correlation, deconvolution

Noise
^^^^^

n-dimensional noise generation:

* White noise
* Pink (1/f) noise
* Brown (1/f²) noise
* Blue noise
* Violet noise
* Gaussian noise
* Poisson noise
* Perlin noise
* Simplex noise
* Worley (cellular) noise
* Fractal Brownian motion
* Value noise

Integral Transforms
^^^^^^^^^^^^^^^^^^^

Transforms between function representations:

* Fourier transforms (DFT, DCT, DST, fractional, short-time)
* Laplace transforms (numerical forward and inverse)
* Hankel transforms
* Mellin transforms
* Radon transforms (forward, inverse, fan-beam, cone-beam)
* Wavelet transforms (CWT, DWT, wavelet packets, scattering transforms)
* Z-transforms
* Hilbert transforms ✓
* Hilbert-Huang transform

Wavelets
^^^^^^^^

Multiresolution analysis and wavelet processing:

* Discrete wavelet transform (1D, 2D, 3D)
* Continuous wavelet transform
* Wavelet packet decomposition
* Maximal overlap DWT (MODWT)
* Stationary wavelet transform
* Wavelet families (Daubechies, Symlets, Coiflets, biorthogonal, Meyer, Morlet, Mexican hat)
* Wavelet-based denoising and thresholding
* Multiresolution analysis

Polynomials
^^^^^^^^^^^

Polynomial arithmetic and manipulation:

* Polynomial evaluation (Horner, compensated Horner)
* Root finding (companion matrix, Laguerre, Jenkins-Traub)
* Polynomial arithmetic (multiplication, division, composition)
* Interpolation (Lagrange, Newton, Chebyshev, barycentric)
* Least squares fitting
* Orthogonal polynomial expansions
* Rational function approximation (Padé, minimax)
* Splines (B-splines, cubic, natural, periodic)

Integration
^^^^^^^^^^^

Numerical integration and differential equation solvers.

Quadrature
""""""""""

* Adaptive quadrature (Gauss-Kronrod, Clenshaw-Curtis)
* Fixed-order quadrature (Gauss-Legendre, Gauss-Laguerre, Gauss-Hermite)
* Oscillatory integrals (Levin, Filon)
* Multidimensional integration (cubature, sparse grids, Monte Carlo)
* Improper integrals (infinite limits, singularities)
* Contour integration

Ordinary Differential Equations
"""""""""""""""""""""""""""""""

* Explicit methods (Euler, Runge-Kutta, multistep)
* Implicit methods (backward Euler, BDF)
* Stiff solvers
* Symplectic integrators
* Adaptive step size control

Boundary Value Problems
"""""""""""""""""""""""

* Shooting methods
* Finite difference methods
* Collocation methods

Differential Algebraic Equations
""""""""""""""""""""""""""""""""

* Index reduction
* Consistent initialization
* DAE solvers

Partial Differential Equations
""""""""""""""""""""""""""""""

* Finite difference stencils
* Spectral methods
* Method of lines

Numerical Optimization
^^^^^^^^^^^^^^^^^^^^^^

General-purpose optimization algorithms.

Unconstrained Optimization
""""""""""""""""""""""""""

* Gradient descent (vanilla, momentum, Nesterov)
* Quasi-Newton methods (BFGS, L-BFGS, SR1)
* Conjugate gradient methods
* Trust region methods
* Newton and Gauss-Newton methods
* Nelder-Mead simplex
* Powell's method
* Coordinate descent

Constrained Optimization
""""""""""""""""""""""""

* Augmented Lagrangian methods
* Sequential quadratic programming (SQP)
* Interior point methods
* Penalty methods
* Projected gradient methods
* Frank-Wolfe algorithm
* ADMM (Alternating Direction Method of Multipliers)

Convex Optimization
"""""""""""""""""""

* Linear programming (simplex, interior point)
* Quadratic programming
* Second-order cone programming (SOCP)
* Semidefinite programming (SDP)
* Geometric programming
* Proximal operators and algorithms (proximal gradient, FISTA, Douglas-Rachford)
* Conic optimization

Combinatorial Optimization
""""""""""""""""""""""""""

* Branch and bound
* Dynamic programming utilities
* Hungarian algorithm
* Traveling salesman heuristics
* Graph optimization (shortest path, maximum flow, minimum cut)
* Assignment problems
* Knapsack problems

Global Optimization
"""""""""""""""""""

* Basin hopping
* Simulated annealing
* Differential evolution
* Particle swarm optimization
* Genetic algorithms
* Bayesian optimization
* Multi-start methods

Root Finding
""""""""""""

* Bisection method
* Brent's method
* Newton-Raphson (scalar and multivariate)
* Secant method
* Ridder's method
* Muller's method
* Fixed-point iteration
* Polynomial root finding (companion matrix, Laguerre, Jenkins-Traub)
* Bracketing utilities

Optimization Test Functions
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Standard benchmark functions for optimization research:

* Unimodal functions (sphere, Rosenbrock ✓, bent cigar, different powers)
* Multimodal functions (Ackley, Rastrigin, Schwefel, Griewank, Levy)
* Separable and non-separable variants
* Constrained test problems (CEC suites)
* Multi-objective test problems (DTLZ, ZDT, WFG)

Probability
^^^^^^^^^^^

Probability distributions and operations:

* Univariate distributions (normal, exponential, gamma, beta, chi-squared, Student's t, F, etc.)
* Multivariate distributions (multivariate normal, Wishart, Dirichlet, multinomial)
* Copulas (Gaussian, Clayton, Frank, Gumbel, Student's t)
* Distribution operations (convolution, mixture, transformation)
* Characteristic functions and moment generating functions
* Quantile functions and inverse CDF
* Random variate generation (rejection sampling, inverse transform, ziggurat)
* Quasi-random sequences (Sobol, Halton, Latin hypercube)

Statistics
^^^^^^^^^^

Statistical inference and analysis.

Descriptive Statistics
""""""""""""""""""""""

* Moments (mean, variance, skewness, kurtosis ✓)
* Quantiles and percentiles
* Order statistics
* L-moments
* Correlation (Pearson, Spearman, Kendall, distance correlation)
* Kernel density estimation

Hypothesis Testing
""""""""""""""""""

* t-tests (one-sample, two-sample, paired)
* Chi-squared tests
* F-tests
* Nonparametric tests (Mann-Whitney, Wilcoxon, Kruskal-Wallis)
* Confidence intervals (parametric, bootstrap, exact)
* Multiple comparison corrections (Bonferroni, Holm, FDR)

Regression
""""""""""

* Ordinary least squares (OLS)
* Weighted and generalized least squares (WLS, GLS)
* Robust regression
* Quantile regression
* Analysis of variance (ANOVA, MANOVA)

Advanced Topics
"""""""""""""""

* Survival analysis (Kaplan-Meier, Cox proportional hazards)
* Time series (ARIMA, spectral analysis, state space models)
* Bayesian inference utilities
* Resampling methods (bootstrap, jackknife, permutation)
* Maximum likelihood estimation utilities
* Method of moments estimation

Information Theory
^^^^^^^^^^^^^^^^^^

Measures of information and coding:

* Entropy (Shannon, Rényi, Tsallis, differential)
* Mutual information (discrete and continuous)
* KL divergence and other f-divergences
* Channel capacity
* Rate-distortion functions
* Source coding bounds
* Entropy estimation (plug-in, nearest neighbor, kernel)

Clustering
^^^^^^^^^^

Clustering algorithms and utilities:

* k-means and k-means++
* k-medoids (PAM)
* DBSCAN and HDBSCAN
* Hierarchical clustering (agglomerative, divisive)
* Spectral clustering
* Gaussian mixture models
* Mean shift
* Cluster validation metrics (silhouette, Davies-Bouldin, Calinski-Harabasz)

Combinatorics
^^^^^^^^^^^^^

Combinatorial algorithms and enumeration:

* Binomial coefficients and multinomial coefficients
* Permutations and combinations
* Integer partitions
* Set partitions
* Catalan numbers and related sequences
* Stirling numbers
* Bell numbers
* Derangements
* Gray codes
* Combinatorial generation (next permutation, next combination)

Compression
^^^^^^^^^^^

Data compression primitives:

* Huffman coding
* Arithmetic coding
* Lempel-Ziv variants
* Run-length encoding
* Burrows-Wheeler transform
* Move-to-front transform
* Dictionary methods
* Transform coding utilities

Distance
^^^^^^^^

Distance and similarity functions:

* Minkowski distances (Euclidean, Manhattan, Chebyshev)
* Mahalanobis distance
* Cosine similarity and distance
* Hamming distance
* Levenshtein (edit) distance
* Jaccard distance
* Hausdorff distance
* Earth mover's distance (Wasserstein)
* Dynamic time warping
* Bregman divergences

Linear Algebra Extensions
^^^^^^^^^^^^^^^^^^^^^^^^^

Advanced linear algebra beyond PyTorch core:

* Matrix functions (exponential, logarithm, square root, arbitrary functions)
* Matrix equations (Sylvester, Lyapunov, Riccati)
* Structured matrices (Toeplitz, Hankel, circulant, banded, sparse patterns)
* Tensor decompositions (Tucker, CP, tensor train, tensor ring)
* Randomized algorithms (randomized SVD, Johnson-Lindenstrauss)
* Matrix nearness problems
* Procrustes problems
* Low-rank approximation

Geometry
^^^^^^^^

Geometric algorithms and spatial data structures:

* Convex hull (2D, 3D, higher dimensions)
* Voronoi diagrams and Delaunay triangulation
* Point location
* Range searching
* Nearest neighbor queries
* Polygon operations (intersection, union, difference, offset)
* Geometric predicates (orientation, in-circle)
* Curve and surface fitting
* Mesh generation and manipulation
* Signed distance functions
* Point cloud processing

Geometric Transforms
""""""""""""""""""""

* Rotation matrices (2D, 3D, arbitrary dimension)
* Euler angles and quaternions
* Axis-angle representation
* Translation and scaling
* Affine transformations
* Projective transformations
* Homogeneous coordinates
* Coordinate system conversions (Cartesian, polar, spherical, cylindrical)

Intersections
"""""""""""""

* Ray-triangle intersection
* Ray-sphere intersection
* Ray-box intersection
* Ray-plane intersection
* Line-line intersection
* Segment-segment intersection
* Polygon-polygon intersection
* Mesh-ray intersection (BVH accelerated)

Space Partitioning
^^^^^^^^^^^^^^^^^^

Hierarchical spatial data structures:

* k-d trees
* Ball trees
* R-trees
* Octrees and quadtrees
* BSP trees
* BVH (bounding volume hierarchies)
* Uniform grids
* Spatial hashing

Graphics
^^^^^^^^

Rendering and graphics primitives.

Color
"""""

* Color space conversions (sRGB, HSV, HSL, LAB, XYZ, LUV, OkLAB)
* Gamma correction and transfer functions
* White balance
* Color difference metrics (Delta E)
* Chromatic adaptation

Shading
"""""""

* BRDF models (phong, blinn_phong, cook_torrance, ggx)
* Environment mapping
* SH lighting (using spherical harmonics from special_functions)
* Ambient occlusion
* Shadow mapping utilities

Image Operations
""""""""""""""""

* Tone mapping operators (Reinhard, ACES, filmic)
* Image filtering (bilateral, guided, non-local means)
* Morphological operations (erosion, dilation, opening, closing)
* Distance transforms
* Connected components

Finite Element Methods
^^^^^^^^^^^^^^^^^^^^^^

Finite element analysis utilities:

* Shape functions (Lagrange, Hermite, hierarchical)
* Element matrices (stiffness, mass, damping)
* Mesh generation (structured, unstructured)
* Assembly routines
* Boundary condition application
* Gaussian quadrature nodes and weights
* Isoparametric mapping
* Error estimation and adaptivity

Graph Theory
^^^^^^^^^^^^

Graph algorithms and data structures:

* Shortest paths (Dijkstra, Bellman-Ford, Floyd-Warshall, A*)
* Minimum spanning trees (Prim, Kruskal, Borůvka)
* Connected components (strongly, weakly)
* Topological sorting
* Maximum flow and minimum cut
* Bipartite matching
* Graph coloring
* Centrality measures (betweenness, closeness, eigenvector, PageRank)
* Community detection
* Graph isomorphism
* Cycle detection

Interpolation and Approximation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Function approximation techniques:

* Polynomial interpolation
* Spline interpolation (cubic, B-spline, NURBS)
* Radial basis function interpolation
* Kriging / Gaussian process interpolation
* Scattered data interpolation
* Multivariate interpolation
* Chebyshev approximation
* Rational approximation

Number Theory
^^^^^^^^^^^^^

Number-theoretic algorithms:

* Prime generation and testing (sieve of Eratosthenes, Miller-Rabin)
* Prime factorization
* Greatest common divisor and least common multiple
* Modular arithmetic (exponentiation, inverse, Chinese remainder theorem)
* Euler's totient function
* Möbius function
* Divisor functions
* Continued fractions
* Diophantine equations

Numerical Differentiation
^^^^^^^^^^^^^^^^^^^^^^^^^

Derivative approximation:

* Finite difference formulas (forward, backward, central, arbitrary order)
* Complex step differentiation
* Richardson extrapolation
* Automatic differentiation utilities (beyond autograd)
* Jacobian and Hessian computation utilities
