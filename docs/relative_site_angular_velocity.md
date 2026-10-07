Relative site angular velocity uses the difference of world angular velocities,
projected into the reference site's axes: `R_A.T @ (omega_B - omega_A)`. `R_A`
is MuJoCo's local-to-world reference `site_xmat`. Angular components have units
rad/s and precede the linear components in the six-dimensional feature.

The previous implementation rotated one world angular velocity by a relative
site orientation before subtracting the other world angular velocity. Two
sites rigidly rotating together could therefore produce a nonzero feature.
The corrected angular feature agrees with MuJoCo's `frameangvel` sensor when
the reference site is specified.

This correction changes relative angular velocity rewards, metrics, and legacy
goal or observation encoders that call `calculate_relative_site_quantities`.
The newer tracking feature encoders compute their velocities independently.
Checkpoint array layouts and public function signatures remain compatible,
but rollout behavior and training resume are not numerically equivalent when
an affected encoder or reward is enabled. Preserve the original code revision
when reproducing a checkpoint's historical evaluation; record this change for
new training or comparisons.

Supported trajectory caches store raw kinematics, including site rotations and
body spatial velocities. Relative features are computed at use time, so those
caches do not require regeneration. Externally stored derived angular features
must be recomputed before comparison with the corrected implementation.

The linear component retains its historical `R_A @ (v_A - v_B)` definition.
It is not the derivative of position in a rotating reference frame. Changing
its frame, sign, or reference-motion correction is a separate behavior change.

Regression coverage uses an asset-free MuJoCo model with a free root, a moving
hinge, and rotated sites. It checks rigid co-rotation, nonzero sensor agreement,
global rotation invariance, dense and reordered sparse trajectory data, NumPy,
JAX eager execution, JIT compilation, and JIT with vmap.
