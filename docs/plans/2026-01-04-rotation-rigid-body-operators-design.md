# Rotation and Rigid-Body Operators for torchscience.geometry.transform

**Date:** 2026-01-04
**Status:** Design Complete

## Overview

Add comprehensive rotation (SO(3)) and rigid-body transformation (SE(3)) operators to `torchscience.geometry.transform`. These operators serve robotics/kinematics, graphics/rendering, physics simulation, computer vision, and general-purpose transformation needs.

## Design Decisions

- **Representations:** Quaternions, rotation matrices, axis-angle, Euler angles (all 24 conventions), Rodrigues/rotation vectors
- **Operations:** Conversions, composition, inversion, application to points, interpolation (slerp, squad, sclerp), derivatives, log/exp maps
- **Rigid body:** Full SE(3) algebra with exp/log, twist vectors, adjoint representation
- **API style:** Tensorclass containers with functional operations
- **Dual quaternions:** Supported as alternative SE(3) representation
- **Euler conventions:** All 24 (12 intrinsic uppercase, 12 extrinsic lowercase)
- **Quaternion convention:** Scalar-first (wxyz) matching PyTorch3D, scipy, Eigen

## Tensorclass Containers

### SO(3) Rotation Representations

```python
@tensorclass
class Quaternion:
    wxyz: Tensor  # shape (..., 4), scalar-first convention [w, x, y, z]

@tensorclass
class RotationMatrix:
    matrix: Tensor  # shape (..., 3, 3)

@tensorclass
class AxisAngle:
    axis: Tensor   # shape (..., 3), unit vector
    angle: Tensor  # shape (...,), radians

@tensorclass
class EulerAngles:
    angles: Tensor      # shape (..., 3), radians
    convention: str     # e.g., "ZYX", "XYZ", "ZXZ" (uppercase=intrinsic, lowercase=extrinsic)

@tensorclass
class RotationVector:
    rotvec: Tensor  # shape (..., 3), axis * angle (Rodrigues vector)
```

### SE(3) Rigid Transform Representations

```python
@tensorclass
class RigidTransform:
    rotation: Quaternion  # SO(3) rotation
    translation: Tensor   # shape (..., 3)

@tensorclass
class DualQuaternion:
    real: Tensor  # shape (..., 4), rotation part
    dual: Tensor  # shape (..., 4), translation part
```

## SO(3) Conversion Functions

All conversions follow the pattern `<source>_to_<target>`:

```python
# Quaternion conversions
quaternion_to_matrix(q: Quaternion) -> RotationMatrix
quaternion_to_axis_angle(q: Quaternion) -> AxisAngle
quaternion_to_euler(q: Quaternion, convention: str = "ZYX") -> EulerAngles
quaternion_to_rotation_vector(q: Quaternion) -> RotationVector

# Rotation matrix conversions
matrix_to_quaternion(m: RotationMatrix) -> Quaternion
matrix_to_axis_angle(m: RotationMatrix) -> AxisAngle
matrix_to_euler(m: RotationMatrix, convention: str = "ZYX") -> EulerAngles
matrix_to_rotation_vector(m: RotationMatrix) -> RotationVector

# Axis-angle conversions
axis_angle_to_quaternion(aa: AxisAngle) -> Quaternion
axis_angle_to_matrix(aa: AxisAngle) -> RotationMatrix
axis_angle_to_euler(aa: AxisAngle, convention: str = "ZYX") -> EulerAngles
axis_angle_to_rotation_vector(aa: AxisAngle) -> RotationVector

# Euler conversions
euler_to_quaternion(e: EulerAngles) -> Quaternion
euler_to_matrix(e: EulerAngles) -> RotationMatrix
euler_to_axis_angle(e: EulerAngles) -> AxisAngle
euler_to_rotation_vector(e: EulerAngles) -> RotationVector

# Rotation vector conversions
rotation_vector_to_quaternion(rv: RotationVector) -> Quaternion
rotation_vector_to_matrix(rv: RotationVector) -> RotationMatrix
rotation_vector_to_axis_angle(rv: RotationVector) -> AxisAngle
rotation_vector_to_euler(rv: RotationVector, convention: str = "ZYX") -> EulerAngles
```

**Implementation note:** Not all conversions need direct implementations. Route through quaternion internally where efficient.

## SO(3) Operations

### Composition and Inversion

```python
# Compose rotations (R1 then R2, i.e., R2 @ R1 in matrix form)
quaternion_multiply(q1: Quaternion, q2: Quaternion) -> Quaternion
matrix_multiply(m1: RotationMatrix, m2: RotationMatrix) -> RotationMatrix

# Inverse rotation
quaternion_inverse(q: Quaternion) -> Quaternion
matrix_inverse(m: RotationMatrix) -> RotationMatrix  # transpose for orthogonal

# Normalize (project back to SO(3) after numerical drift)
quaternion_normalize(q: Quaternion) -> Quaternion
matrix_normalize(m: RotationMatrix) -> RotationMatrix  # SVD orthogonalization
```

### Application to Points/Vectors

```python
# Rotate points/vectors: shape (..., 3) -> (..., 3)
quaternion_apply(q: Quaternion, points: Tensor) -> Tensor
matrix_apply(m: RotationMatrix, points: Tensor) -> Tensor
axis_angle_apply(aa: AxisAngle, points: Tensor) -> Tensor  # Rodrigues formula
rotation_vector_apply(rv: RotationVector, points: Tensor) -> Tensor
```

### Interpolation

```python
# Spherical linear interpolation
quaternion_slerp(q1: Quaternion, q2: Quaternion, t: Tensor) -> Quaternion

# Spherical quadrangle interpolation (smooth spline through 4 quaternions)
quaternion_squad(q0: Quaternion, q1: Quaternion, q2: Quaternion, q3: Quaternion, t: Tensor) -> Quaternion
```

### Exponential/Logarithm Maps

```python
# so(3) <-> SO(3) mappings
quaternion_exp(tangent: Tensor) -> Quaternion      # tangent shape (..., 3)
quaternion_log(q: Quaternion) -> Tensor            # returns (..., 3)
matrix_exp(skew: Tensor) -> RotationMatrix         # skew shape (..., 3, 3)
matrix_log(m: RotationMatrix) -> Tensor            # returns (..., 3, 3) skew-symmetric
```

## SE(3) Rigid Transform Operations

### Constructors and Conversions

```python
# Create rigid transform from components
rigid_transform(rotation: Quaternion, translation: Tensor) -> RigidTransform
rigid_transform_from_matrix(matrix: Tensor) -> RigidTransform  # 4x4 homogeneous
rigid_transform_to_matrix(rt: RigidTransform) -> Tensor        # returns (..., 4, 4)

# Dual quaternion conversions
rigid_transform_to_dual_quaternion(rt: RigidTransform) -> DualQuaternion
dual_quaternion_to_rigid_transform(dq: DualQuaternion) -> RigidTransform
```

### Composition and Inversion

```python
# Compose transforms: T1 then T2
rigid_transform_compose(t1: RigidTransform, t2: RigidTransform) -> RigidTransform
dual_quaternion_multiply(dq1: DualQuaternion, dq2: DualQuaternion) -> DualQuaternion

# Inverse transform
rigid_transform_inverse(rt: RigidTransform) -> RigidTransform
dual_quaternion_inverse(dq: DualQuaternion) -> DualQuaternion

# Normalize
dual_quaternion_normalize(dq: DualQuaternion) -> DualQuaternion
```

### Application to Points

```python
# Transform points: shape (..., 3) -> (..., 3)
rigid_transform_apply(rt: RigidTransform, points: Tensor) -> Tensor
dual_quaternion_apply(dq: DualQuaternion, points: Tensor) -> Tensor
```

### Interpolation

```python
# Linear interpolation (decoupled rotation/translation)
rigid_transform_lerp(t1: RigidTransform, t2: RigidTransform, t: Tensor) -> RigidTransform

# Screw-linear interpolation (constant screw motion)
dual_quaternion_sclerp(dq1: DualQuaternion, dq2: DualQuaternion, t: Tensor) -> DualQuaternion
```

## SE(3) Lie Algebra Operations

### Exponential and Logarithm Maps

```python
# se(3) -> SE(3): twist vector to rigid transform
rigid_transform_exp(twist: Tensor) -> RigidTransform  # twist shape (..., 6)

# SE(3) -> se(3): rigid transform to twist vector
rigid_transform_log(rt: RigidTransform) -> Tensor     # returns (..., 6)
```

### Twist Vector Utilities

```python
# Twist vector layout: [angular (3), linear (3)] following robotics convention

# Convert twist to 4x4 matrix representation (se(3) Lie algebra element)
twist_to_matrix(twist: Tensor) -> Tensor    # (..., 6) -> (..., 4, 4)
matrix_to_twist(matrix: Tensor) -> Tensor   # (..., 4, 4) -> (..., 6)

# Hat and vee operators for so(3) (skew-symmetric)
skew_symmetric(v: Tensor) -> Tensor         # (..., 3) -> (..., 3, 3)
skew_symmetric_inverse(m: Tensor) -> Tensor # (..., 3, 3) -> (..., 3)
```

### Adjoint Representation

```python
# 6x6 adjoint matrix for transforming twists between frames
rigid_transform_adjoint(rt: RigidTransform) -> Tensor  # returns (..., 6, 6)

# Apply adjoint to twist: Ad_T @ twist
rigid_transform_adjoint_apply(rt: RigidTransform, twist: Tensor) -> Tensor
```

### Derivatives (Jacobians)

```python
# Left/right Jacobians of SO(3) exponential map
rotation_exp_left_jacobian(rotvec: Tensor) -> Tensor   # (..., 3) -> (..., 3, 3)
rotation_exp_right_jacobian(rotvec: Tensor) -> Tensor  # (..., 3) -> (..., 3, 3)

# Inverse Jacobians
rotation_exp_left_jacobian_inverse(rotvec: Tensor) -> Tensor
rotation_exp_right_jacobian_inverse(rotvec: Tensor) -> Tensor
```

## File Organization

### Python API

```
src/torchscience/geometry/transform/
├── __init__.py
├── _reflect.py                    # existing
├── _refract.py                    # existing
├── _quaternion.py                 # Quaternion tensorclass + all quaternion_* functions
├── _rotation_matrix.py            # RotationMatrix tensorclass + all matrix_* functions
├── _axis_angle.py                 # AxisAngle tensorclass + axis_angle_* functions
├── _euler_angles.py               # EulerAngles tensorclass + euler_* functions
├── _rotation_vector.py            # RotationVector tensorclass + rotation_vector_* functions
├── _rigid_transform.py            # RigidTransform tensorclass + rigid_transform_* functions
├── _dual_quaternion.py            # DualQuaternion tensorclass + dual_quaternion_* functions
├── _twist.py                      # twist_*, skew_symmetric_* functions
└── _jacobian.py                   # rotation_exp_*_jacobian functions
```

### C++ Kernels

```
src/torchscience/csrc/kernel/geometry/transform/
├── quaternion_multiply.h
├── quaternion_inverse.h
├── quaternion_apply.h
├── quaternion_slerp.h
├── quaternion_exp.h
├── quaternion_log.h
├── quaternion_to_matrix.h
├── matrix_to_quaternion.h
├── euler_to_quaternion.h          # handles all 24 conventions
├── quaternion_to_euler.h
├── rigid_transform_exp.h
├── rigid_transform_log.h
├── dual_quaternion_multiply.h
├── dual_quaternion_sclerp.h
└── ... (backward kernels for each)
```

Backend registrations in `cpu/`, `meta/`, `autograd/`, `autocast/` directories following existing patterns.

## Implementation Phases

### Phase 1 - Core Quaternion Operations (highest value)

- `Quaternion` tensorclass
- `quaternion_multiply`, `quaternion_inverse`, `quaternion_normalize`
- `quaternion_apply`
- `quaternion_to_matrix`, `matrix_to_quaternion`
- `quaternion_slerp`

### Phase 2 - Full SO(3) Representations

- `RotationMatrix`, `AxisAngle`, `RotationVector`, `EulerAngles` tensorclasses
- All conversion functions (routing through quaternion where efficient)
- `quaternion_exp`, `quaternion_log`
- `quaternion_squad`

### Phase 3 - SE(3) Rigid Transforms

- `RigidTransform` tensorclass
- `rigid_transform_compose`, `rigid_transform_inverse`, `rigid_transform_apply`
- `rigid_transform_exp`, `rigid_transform_log`
- `rigid_transform_adjoint`, `rigid_transform_adjoint_apply`

### Phase 4 - Dual Quaternions and Jacobians

- `DualQuaternion` tensorclass
- `dual_quaternion_multiply`, `dual_quaternion_sclerp`, `dual_quaternion_apply`
- `rotation_exp_left_jacobian`, `rotation_exp_right_jacobian` and inverses

## Testing Strategy

- Forward correctness against `scipy.spatial.transform.Rotation`
- `torch.autograd.gradcheck` for all operations
- `torch.autograd.gradgradcheck` for second-order gradients
- Round-trip tests (e.g., `matrix_to_quaternion(quaternion_to_matrix(q)) ≈ q`)
- Singularity handling (gimbal lock for Euler, identity rotation for log)
- Batched operations with broadcasting