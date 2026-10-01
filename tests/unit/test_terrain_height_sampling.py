"""
Unit tests for HeightMatrix observation and terrain height sampling.

Tests cover:
1. Observation shape and dimensions
2. Grid rotation with robot heading
3. Height sampling accuracy for RoughTerrain
4. Static terrain returns zeros
5. Bilinear interpolation correctness
6. Coordinate mapping at grid points
7. JIT vectorized sampling parity
8. HeightMatrix relative height assembly
9. Heightfield conversion (isaac_hf_to_mujoco_hf)
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from loco_mujoco.core.observations.base import HeightMatrix, ObservationType
from loco_mujoco.core.terrain.rough import RoughTerrain


class TestHeightMatrixBasics:
    """Test basic HeightMatrix functionality."""

    def test_height_matrix_registered_in_observation_type(self):
        """Verify HeightMatrix is registered in ObservationType."""
        assert hasattr(ObservationType, "HeightMatrix")
        assert ObservationType.HeightMatrix is HeightMatrix

    def test_height_matrix_dimension_calculation(self):
        """Verify dim is correctly calculated from grid size."""
        obs = HeightMatrix("test", grid_rows=10, grid_cols=10)
        assert obs.dim == 100

        obs = HeightMatrix("test", grid_rows=5, grid_cols=8)
        assert obs.dim == 40

    def test_height_matrix_default_parameters(self):
        """Verify default parameters are set correctly."""
        obs = HeightMatrix("test")
        assert obs.grid_rows == 10
        assert obs.grid_cols == 10
        assert obs.grid_resolution == 0.1
        assert obs.grid_forward_offset == 0.0
        assert obs.body_name is None


class TestQuatToYaw:
    """Test quaternion to yaw conversion."""

    def test_identity_quaternion_gives_zero_yaw(self):
        """Identity quaternion (w=1, x=y=z=0) should give yaw=0."""
        obs = HeightMatrix("test")
        quat = jnp.array([1.0, 0.0, 0.0, 0.0])  # wxyz format
        yaw = obs._quat_to_yaw(quat, jnp)
        assert jnp.abs(yaw) < 1e-6

    def test_90_degree_yaw_rotation(self):
        """90 degree rotation about Z axis."""
        obs = HeightMatrix("test")
        # Quaternion for 90 degree rotation about Z: w=cos(45°), z=sin(45°)
        angle = jnp.pi / 2
        quat = jnp.array([jnp.cos(angle / 2), 0.0, 0.0, jnp.sin(angle / 2)])
        yaw = obs._quat_to_yaw(quat, jnp)
        np.testing.assert_allclose(yaw, jnp.pi / 2, atol=1e-6)

    def test_180_degree_yaw_rotation(self):
        """180 degree rotation about Z axis."""
        obs = HeightMatrix("test")
        angle = jnp.pi
        quat = jnp.array([jnp.cos(angle / 2), 0.0, 0.0, jnp.sin(angle / 2)])
        yaw = obs._quat_to_yaw(quat, jnp)
        np.testing.assert_allclose(jnp.abs(yaw), jnp.pi, atol=1e-6)

    def test_numpy_backend(self):
        """Test with numpy backend."""
        obs = HeightMatrix("test")
        quat = np.array([1.0, 0.0, 0.0, 0.0])
        yaw = obs._quat_to_yaw(quat, np)
        assert np.abs(yaw) < 1e-6


class TestGridOffsets:
    """Test grid offset computation."""

    def test_grid_offsets_centered(self):
        """Grid offsets should be centered around (0, 0)."""
        obs = HeightMatrix("test", grid_rows=3, grid_cols=3, grid_resolution=1.0)
        # Manually compute expected offsets
        obs._grid_offsets_np = None
        obs._grid_offsets_jnp = None

        # Simulate _init_from_mj offset computation
        half_rows = (3 - 1) / 2  # = 1.0
        half_cols = (3 - 1) / 2  # = 1.0

        offsets = []
        for i in range(3):
            for j in range(3):
                x_offset = (i - half_rows) * 1.0
                y_offset = (j - half_cols) * 1.0
                offsets.append([x_offset, y_offset])

        expected = np.array(offsets)

        # Check center point is at (0, 0)
        center_idx = 4  # For 3x3 grid, center is index 4
        np.testing.assert_allclose(expected[center_idx], [0.0, 0.0])

        # Check corners
        np.testing.assert_allclose(expected[0], [-1.0, -1.0])  # Top-left
        np.testing.assert_allclose(expected[8], [1.0, 1.0])    # Bottom-right


class TestStaticTerrainSampling:
    """Test height sampling for static terrain."""

    def test_static_terrain_returns_zeros(self):
        """StaticTerrain.sample_heights_at_points should return zeros."""
        from loco_mujoco.core.terrain.static import StaticTerrain

        # Create a minimal mock environment
        class MockEnv:
            pass

        terrain = StaticTerrain(MockEnv())

        x = jnp.array([0.0, 1.0, -1.0, 2.5])
        y = jnp.array([0.0, 0.5, -0.5, 1.0])

        heights = terrain.sample_heights_at_points(x, y, None, None, jnp)

        np.testing.assert_allclose(heights, 0.0, atol=1e-6)

    def test_static_terrain_numpy_backend(self):
        """Test StaticTerrain with numpy backend."""
        from loco_mujoco.core.terrain.static import StaticTerrain

        class MockEnv:
            pass

        terrain = StaticTerrain(MockEnv())

        x = np.array([0.0, 1.0, -1.0])
        y = np.array([0.0, 0.5, -0.5])

        heights = terrain.sample_heights_at_points(x, y, None, None, np)

        np.testing.assert_allclose(heights, 0.0, atol=1e-6)


class TestRoughTerrainSampling:
    """Test height sampling for rough terrain."""

    def test_coordinate_mapping_bounds(self):
        """Test that coordinate mapping handles boundary conditions."""
        from loco_mujoco.core.terrain.rough import RoughTerrain
        from flax import struct

        @struct.dataclass
        class MockTerrainState:
            height_field_raw: jnp.ndarray

        @struct.dataclass
        class MockCarry:
            terrain_state: MockTerrainState

        class MockEnv:
            root_free_joint_xml_name = "root"
            _model = None

        # Create terrain with known parameters
        terrain = RoughTerrain.__new__(RoughTerrain)
        terrain.hfield_length = 80
        terrain.hfield_half_length_in_meters = 4.0
        terrain.hfield_size = (4, 4, 30.0, 0.125)
        terrain.elevation_z = 30.0  # size[2] - maximum height scale
        terrain.geom_z = -0.06  # geom position z-coordinate

        # Create a simple height field (all zeros = flat at lowest point)
        height_field = jnp.zeros((80, 80))
        carry = MockCarry(terrain_state=MockTerrainState(height_field_raw=height_field.ravel()))

        # Test center point (world 0,0 -> hfield center)
        x = jnp.array([0.0])
        y = jnp.array([0.0])
        heights = terrain.sample_heights_at_points(x, y, None, carry, jnp)

        # For flat terrain (normalized = 0), height = geom_z + 0 * elevation_z = geom_z
        # Note: base_z (size[3]) is downward thickness, NOT added to surface height
        expected_height = terrain.geom_z  # -0.06
        np.testing.assert_allclose(heights[0], expected_height, atol=1e-6)

    def test_bilinear_interpolation(self):
        """Test that bilinear interpolation works correctly."""
        from loco_mujoco.core.terrain.rough import RoughTerrain
        from flax import struct

        @struct.dataclass
        class MockTerrainState:
            height_field_raw: jnp.ndarray

        @struct.dataclass
        class MockCarry:
            terrain_state: MockTerrainState

        # Create terrain with known parameters
        terrain = RoughTerrain.__new__(RoughTerrain)
        terrain.hfield_length = 4  # Small grid for testing
        terrain.hfield_half_length_in_meters = 2.0  # -2 to +2 meters
        terrain.hfield_size = (2, 2, 1.0, 0.0)  # Simplified for testing
        terrain.elevation_z = 1.0  # Simple scaling for easy verification
        terrain.geom_z = 0.0  # No offset for simplicity

        # Create height field with known values
        # hfield[0,0]=0, hfield[0,3]=0.3, hfield[3,0]=0.3, hfield[3,3]=0.6
        height_field = jnp.array([
            [0.0, 0.1, 0.2, 0.3],
            [0.1, 0.2, 0.3, 0.4],
            [0.2, 0.3, 0.4, 0.5],
            [0.3, 0.4, 0.5, 0.6],
        ])
        carry = MockCarry(terrain_state=MockTerrainState(height_field_raw=height_field.ravel()))

        # Sample at center (should interpolate)
        x = jnp.array([0.0])
        y = jnp.array([0.0])
        heights = terrain.sample_heights_at_points(x, y, None, carry, jnp)

        # Center of 4x4 grid at world (0,0) should map to hfield index (1.5, 1.5)
        # Bilinear interpolation of h[1,1]=0.2, h[1,2]=0.3, h[2,1]=0.3, h[2,2]=0.4
        # = 0.25 * (0.2 + 0.3 + 0.3 + 0.4) = 0.3
        # With elevation_z=1.0 and geom_z=0: height = 0.3
        assert heights.shape == (1,)
        np.testing.assert_allclose(heights[0], 0.3, atol=1e-6)

    def test_axis_not_swapped(self):
        """Test that X and Y axes are not swapped using asymmetric height field."""
        from loco_mujoco.core.terrain.rough import RoughTerrain
        from flax import struct

        @struct.dataclass
        class MockTerrainState:
            height_field_raw: jnp.ndarray

        @struct.dataclass
        class MockCarry:
            terrain_state: MockTerrainState

        terrain = RoughTerrain.__new__(RoughTerrain)
        terrain.hfield_length = 8
        terrain.hfield_half_length_in_meters = 4.0
        terrain.elevation_z = 1.0
        terrain.geom_z = 0.0

        # Height field varies ONLY along X (first index): h[i, j] = i / 7.0
        # This creates a gradient only in the X direction
        height_field = jnp.zeros((8, 8))
        for i in range(8):
            height_field = height_field.at[i, :].set(i / 7.0)
        carry = MockCarry(terrain_state=MockTerrainState(height_field_raw=height_field.ravel()))

        # Moving along world X should change height significantly
        x_positions = jnp.array([-3.0, 0.0, 3.0])
        y_fixed = jnp.array([0.0, 0.0, 0.0])
        heights_x = terrain.sample_heights_at_points(x_positions, y_fixed, None, carry, jnp)

        # Check significant variation along X (should be > 0.2 for this gradient)
        x_variation = float(heights_x[2] - heights_x[0])
        assert x_variation > 0.2, f"X variation too small: {x_variation}, heights: {heights_x}"

        # Moving along world Y should NOT change height
        x_fixed = jnp.array([0.0, 0.0, 0.0])
        y_positions = jnp.array([-3.0, 0.0, 3.0])
        heights_y = terrain.sample_heights_at_points(x_fixed, y_positions, None, carry, jnp)

        # Check no variation along Y (should be < 1e-5)
        y_variation = float(jnp.max(heights_y) - jnp.min(heights_y))
        assert y_variation < 1e-5, f"Y variation too large: {y_variation}, heights: {heights_y}"

    def test_boundary_clip_behavior(self):
        """Test that out-of-bounds coordinates are clipped correctly."""
        from loco_mujoco.core.terrain.rough import RoughTerrain
        from flax import struct

        @struct.dataclass
        class MockTerrainState:
            height_field_raw: jnp.ndarray

        @struct.dataclass
        class MockCarry:
            terrain_state: MockTerrainState

        terrain = RoughTerrain.__new__(RoughTerrain)
        terrain.hfield_length = 8
        terrain.hfield_half_length_in_meters = 4.0
        terrain.elevation_z = 1.0
        terrain.geom_z = 0.0

        height_field = jnp.full((8, 8), 0.5)
        carry = MockCarry(terrain_state=MockTerrainState(height_field_raw=height_field.ravel()))

        # Sample outside boundaries in both X and Y
        x_outside = jnp.array([-5.0, 5.0, 0.0])
        y_outside = jnp.array([0.0, 0.0, 6.0])
        heights_outside = terrain.sample_heights_at_points(x_outside, y_outside, None, carry, jnp)

        # Should not produce NaN/Inf
        assert not jnp.any(jnp.isnan(heights_outside)), "Should not produce NaN"
        assert not jnp.any(jnp.isinf(heights_outside)), "Should not produce Inf"

        # Clipped X values should match boundary values
        x_boundary = jnp.array([-4.0, 4.0])
        y_boundary = jnp.array([0.0, 0.0])
        heights_x_boundary = terrain.sample_heights_at_points(x_boundary, y_boundary, None, carry, jnp)
        np.testing.assert_allclose(heights_outside[0], heights_x_boundary[0], atol=1e-5)
        np.testing.assert_allclose(heights_outside[1], heights_x_boundary[1], atol=1e-5)

        # Clipped Y value should match boundary value
        x_y_boundary = jnp.array([0.0])
        y_y_boundary = jnp.array([4.0])
        heights_y_boundary = terrain.sample_heights_at_points(x_y_boundary, y_y_boundary, None, carry, jnp)
        np.testing.assert_allclose(heights_outside[2], heights_y_boundary[0], atol=1e-5)

    def test_world_to_index_mapping_exact_grid_points(self):
        """Test that world coordinates map to the expected hfield indices at grid points."""
        from loco_mujoco.core.terrain.rough import RoughTerrain
        from flax import struct

        @struct.dataclass
        class MockTerrainState:
            height_field_raw: jnp.ndarray

        @struct.dataclass
        class MockCarry:
            terrain_state: MockTerrainState

        terrain = RoughTerrain.__new__(RoughTerrain)
        terrain.hfield_length = 5
        terrain.hfield_half_length_in_meters = 2.0
        terrain.elevation_z = 1.0
        terrain.geom_z = 0.0

        height_field = jnp.arange(25, dtype=jnp.float32).reshape(5, 5)
        carry = MockCarry(terrain_state=MockTerrainState(height_field_raw=height_field.ravel()))

        indices = jnp.array([[0, 0], [2, 3], [4, 1]])
        denom = terrain.hfield_length - 1
        x_world = (
            indices[:, 0] / denom * (2 * terrain.hfield_half_length_in_meters)
            - terrain.hfield_half_length_in_meters
        )
        y_world = (
            indices[:, 1] / denom * (2 * terrain.hfield_half_length_in_meters)
            - terrain.hfield_half_length_in_meters
        )

        heights = terrain.sample_heights_at_points(x_world, y_world, None, carry, jnp)
        expected = height_field[indices[:, 0], indices[:, 1]]
        np.testing.assert_allclose(heights, expected, atol=1e-6)

    def test_jitted_vectorized_sampling_matches_eager(self):
        """Test JIT-compiled vectorized sampling matches eager mode."""
        from loco_mujoco.core.terrain.rough import RoughTerrain
        from flax import struct

        @struct.dataclass
        class MockTerrainState:
            height_field_raw: jnp.ndarray

        @struct.dataclass
        class MockCarry:
            terrain_state: MockTerrainState

        terrain = RoughTerrain.__new__(RoughTerrain)
        terrain.hfield_length = 8
        terrain.hfield_half_length_in_meters = 4.0
        terrain.elevation_z = 1.0
        terrain.geom_z = 0.0

        height_field = (jnp.arange(64, dtype=jnp.float32).reshape(8, 8) / 10.0)
        height_field_raw = height_field.ravel()

        x_positions = jnp.array([-3.5, 0.0, 2.1, 3.9])
        y_positions = jnp.array([3.5, -1.2, 0.0, -3.9])

        def sample(x, y, height_raw):
            carry = MockCarry(terrain_state=MockTerrainState(height_field_raw=height_raw))
            return terrain.sample_heights_at_points(x, y, None, carry, jnp)

        expected = sample(x_positions, y_positions, height_field_raw)
        actual = jax.jit(sample)(x_positions, y_positions, height_field_raw)

        np.testing.assert_allclose(actual, expected, atol=1e-6)


class TestGridRotation:
    """Test that grid rotates with robot heading."""

    def test_grid_rotation_90_degrees(self):
        """Grid should rotate 90 degrees when robot faces +Y."""
        obs = HeightMatrix("test", grid_rows=3, grid_cols=3, grid_resolution=1.0)

        # Create grid offsets
        half_rows = 1.0
        half_cols = 1.0
        offsets = []
        for i in range(3):
            for j in range(3):
                offsets.append([(i - half_rows), (j - half_cols)])
        obs._grid_offsets_jnp = jnp.array(offsets)

        # Point that was at (1, 0) in local frame (forward)
        # After 90 degree rotation should be at (0, 1) in world frame
        yaw = jnp.pi / 2
        cos_yaw = jnp.cos(yaw)
        sin_yaw = jnp.sin(yaw)

        # Original forward point (row=2, col=1) -> offset (1, 0)
        forward_idx = 7  # row=2, col=1 in flattened 3x3
        local_x = obs._grid_offsets_jnp[forward_idx, 0]  # = 1.0
        local_y = obs._grid_offsets_jnp[forward_idx, 1]  # = 0.0

        rotated_x = cos_yaw * local_x - sin_yaw * local_y
        rotated_y = sin_yaw * local_x + cos_yaw * local_y

        # After 90 degree rotation, (1, 0) -> (0, 1)
        np.testing.assert_allclose(rotated_x, 0.0, atol=1e-6)
        np.testing.assert_allclose(rotated_y, 1.0, atol=1e-6)


class TestHeightMatrixGetObs:
    """Test HeightMatrix observation assembly."""

    def test_get_obs_relative_heights_and_rotation(self):
        """HeightMatrix should rotate the grid and return relative heights."""
        from types import SimpleNamespace

        obs = HeightMatrix("test", grid_rows=2, grid_cols=1, grid_resolution=1.0)
        obs._body_id = 0
        obs._free_jnt_qpos_id = np.array([0, 1, 2, 3, 4, 5, 6])
        obs._grid_offsets_np = np.array([[-0.5, 0.0], [0.5, 0.0]])
        obs._grid_offsets_jnp = jnp.array([[-0.5, 0.0], [0.5, 0.0]])

        class MockTerrain:
            def sample_heights_at_points(self, x, y, model, carry, backend):
                return x + 2 * y

        env = SimpleNamespace(_terrain=MockTerrain())

        robot_pos = jnp.array([1.0, 2.0, 3.0])
        yaw = jnp.pi / 2
        quat = jnp.array([jnp.cos(yaw / 2), 0.0, 0.0, jnp.sin(yaw / 2)])
        qpos = jnp.zeros(7)
        qpos = qpos.at[3:7].set(quat)

        data = SimpleNamespace(
            xpos=jnp.array([robot_pos]),
            qpos=qpos,
        )

        heights, _ = obs.get_obs_and_update_state(env, None, data, None, jnp)
        expected = jnp.array([1.0, 3.0])
        np.testing.assert_allclose(heights, expected, atol=1e-6)


class TestHeightfieldConversion:
    """Test heightfield conversion from Isaac format to MuJoCo format."""

    @staticmethod
    def _make_terrain_with_scaling(scale: float = 30.0) -> RoughTerrain:
        terrain = RoughTerrain.__new__(RoughTerrain)
        terrain.mujoco_height_scaling = scale
        return terrain

    def test_isaac_hf_to_mujoco_hf_shifts_and_scales(self):
        """Test that conversion shifts to non-negative and scales correctly."""
        terrain = self._make_terrain_with_scaling()

        isaac_hf = np.array([[-0.05, 0.0, 0.05]], dtype=np.float32)
        converted = terrain.isaac_hf_to_mujoco_hf(isaac_hf, np)

        expected = (isaac_hf + 0.05) / 30.0
        expected = expected.reshape(-1)

        np.testing.assert_allclose(converted, expected, atol=1e-6)

    def test_isaac_hf_to_mujoco_hf_non_negative_and_flattened(self):
        """Test that output is non-negative and flattened."""
        terrain = self._make_terrain_with_scaling()

        isaac_hf = np.array([[-0.02, -0.01], [0.0, 0.03]], dtype=np.float32)
        converted = terrain.isaac_hf_to_mujoco_hf(isaac_hf, np)

        assert converted.ndim == 1
        assert converted.shape[0] == isaac_hf.size
        assert np.min(converted) >= 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
