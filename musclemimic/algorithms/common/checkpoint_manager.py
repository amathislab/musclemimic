import os
import shutil
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

try:
    import orbax.checkpoint as ocp

    ORBAX_AVAILABLE = True
    ORBAX_VERSION = getattr(ocp, "__version__", "unknown")
except ImportError:
    ORBAX_AVAILABLE = False
    ocp = None
    ORBAX_VERSION = None


@dataclass
class CheckpointMetadata:
    """Metadata associated with a checkpoint (schema v2.2)."""

    # Progress tracking
    step: int  # Optimizer step count
    update_number: int  # Update count (rollout iterations)
    global_timestep: int  # Total environment steps
    target_global_timestep: int  # Absolute training budget for this run
    learning_rate: float

    # Training configuration (required for robust resume)
    num_envs: int
    num_steps: int
    num_minibatches: int
    update_epochs: int

    # Metadata
    schema_version: str = "2.2"
    algo_version: str = "PPOJax_v1"
    backend: str = "warp"
    env_name: str = ""
    wandb_run_id: str = ""


class CheckpointFormat:
    """Supported checkpoint formats."""

    PICKLE = "pickle"
    ORBAX = "orbax"


def checkpoint_path_for_step(checkpoint_dir: str | Path, step: int) -> str:
    """Return the canonical Orbax directory for one checkpoint step."""
    return str(Path(checkpoint_dir).resolve() / f"checkpoint_{int(step)}")


def _is_zero_size_array(x: Any) -> bool:
    shape = getattr(x, "shape", None)
    if shape is None or getattr(x, "dtype", None) is None:
        return False
    return any(int(dim) == 0 for dim in tuple(shape))


def replace_zero_size_arrays(tree: Any) -> Any:
    """Orbax cannot write empty arrays; they carry no values, so write a placeholder."""
    import jax
    import jax.numpy as jnp

    def _replace(x):
        return jnp.zeros((1,), dtype=x.dtype) if _is_zero_size_array(x) else x

    return jax.tree_util.tree_map(_replace, tree, is_leaf=lambda x: x is None or _is_zero_size_array(x))


def restore_zero_size_arrays_from_target(tree: Any, target: Any | None) -> Any:
    """Put empty arrays back using the typed restore target."""
    if target is None:
        return tree

    import jax
    import jax.numpy as jnp

    def _restore(x, target_x):
        if _is_zero_size_array(target_x):
            return jnp.zeros(tuple(target_x.shape), dtype=target_x.dtype)
        return x

    return jax.tree_util.tree_map(_restore, tree, target, is_leaf=lambda x: x is None)


class BaseCheckpointManager:
    """Base class for checkpoint management."""

    def __init__(self, checkpoint_dir: str, max_to_keep: int = 5):
        # Use an absolute directory to satisfy Orbax/Tensorstore requirements.
        self.checkpoint_dir = Path(checkpoint_dir).resolve()
        self.max_to_keep = max_to_keep
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def save_checkpoint(
        self, step: int, agent_conf: Any, agent_state: Any, metadata: CheckpointMetadata | None = None
    ) -> str:
        """Save a checkpoint. Must be implemented by subclasses."""
        raise NotImplementedError

    def load_checkpoint(
        self,
        checkpoint_path: str,
        train_state_target: Any | None = None,
    ) -> tuple[Any, Any, CheckpointMetadata]:
        """Load a checkpoint. Must be implemented by subclasses."""
        raise NotImplementedError

    def list_checkpoints(self) -> list:
        """List all available checkpoints."""
        raise NotImplementedError

    def cleanup_old_checkpoints(self, keep_latest: bool = True):
        """Remove old checkpoints, keeping only max_to_keep most recent."""
        checkpoints = self.list_checkpoints()
        if len(checkpoints) <= self.max_to_keep:
            return

        # Sort by step number (assuming checkpoint names contain step info)
        checkpoints.sort(key=self._extract_step_from_name)

        # Keep the most recent max_to_keep checkpoints
        to_remove = checkpoints[: -self.max_to_keep] if keep_latest else checkpoints[self.max_to_keep :]

        for ckpt_path in to_remove:
            try:
                if os.path.isdir(ckpt_path):
                    shutil.rmtree(ckpt_path)
                else:
                    os.remove(ckpt_path)
                print(f"Removed old checkpoint: {ckpt_path}")
            except Exception as e:
                print(f"Warning: Failed to remove checkpoint {ckpt_path}: {e}")

    def _extract_step_from_name(self, checkpoint_path: str) -> int:
        """Extract step number from checkpoint path/name."""
        path = Path(checkpoint_path)
        if path.is_dir():
            # Directory format: ckpt_000010
            try:
                return int(path.name.split("_")[-1])
            except (ValueError, IndexError):
                return 0
        else:
            # File format: could be various
            try:
                # Try to extract number from filename
                import re

                numbers = re.findall(r"\d+", path.stem)
                return int(numbers[-1]) if numbers else 0
            except (ValueError, IndexError):
                return 0


## Legacy pickle checkpoint manager removed: Orbax-only pipeline


class OrbaxCheckpointManager(BaseCheckpointManager):
    """Modern Orbax-based checkpoint manager for optimal performance."""

    def __init__(
        self,
        checkpoint_dir: str,
        max_to_keep: int = 5,
        save_interval_steps: int = 1,
        async_save: bool = True,
        save_concurrent_gb: int = 4,
        restore_concurrent_gb: int = 4,
    ):
        if not ORBAX_AVAILABLE:
            raise ImportError("Orbax is not available. Install with: pip install orbax-checkpoint")

        super().__init__(checkpoint_dir, max_to_keep)

        # Create Orbax CheckpointManager with modern API
        # Async behavior is controlled via enable_async_checkpointing flag.
        self.async_save = bool(async_save)
        options = ocp.CheckpointManagerOptions(
            max_to_keep=max_to_keep,
            save_interval_steps=save_interval_steps,
            create=True,
            enable_async_checkpointing=self.async_save,
            step_prefix="checkpoint",  # Ensure directory names match returned paths
        )

        # Define item names for composite checkpoints
        item_names = ("train_state", "config", "metadata")
        item_handlers = {
            "train_state": ocp.StandardCheckpointHandler(
                save_concurrent_gb=int(save_concurrent_gb),
                restore_concurrent_gb=int(restore_concurrent_gb),
            ),
            "config": ocp.JsonCheckpointHandler(),
            "metadata": ocp.JsonCheckpointHandler(),
        }

        self.manager = ocp.CheckpointManager(
            str(self.checkpoint_dir),
            options=options,
            item_names=item_names,
            item_handlers=item_handlers,
        )

    def save_checkpoint(
        self, step: int, agent_conf: Any, agent_state: Any, metadata: CheckpointMetadata | None = None
    ) -> str:
        """Save checkpoint in Orbax format for optimal performance."""
        # Prepare checkpoint data (items) and argument specs (args)
        items = {
            "train_state": self._extract_train_state(agent_state),
            "config": self._extract_config(agent_conf),
        }

        # Prepare metadata (always include, even if empty)
        if metadata:
            metadata_dict = {
                "step": metadata.step,
                "update_number": metadata.update_number,
                "global_timestep": metadata.global_timestep,
                "target_global_timestep": metadata.target_global_timestep,
                "learning_rate": metadata.learning_rate,
                "num_envs": metadata.num_envs,
                "num_steps": metadata.num_steps,
                "num_minibatches": metadata.num_minibatches,
                "update_epochs": metadata.update_epochs,
                "schema_version": metadata.schema_version,
                "algo_version": metadata.algo_version,
                "backend": metadata.backend,
                "env_name": metadata.env_name,
                "wandb_run_id": metadata.wandb_run_id,
            }
        else:
            # Create default metadata
            metadata_dict = {
                "step": step,
                "update_number": 0,
                "global_timestep": 0,
                "learning_rate": 0.0,
                "schema_version": "2.0",
                "algo_version": "PPOJax_v1",
                "backend": "jax",
                "env_name": "",
            }

        # Prepare save args which include the actual items for this Orbax version
        # (StandardSaveArgs/JsonSaveArgs require an `item` argument).
        save_args = ocp.args.Composite(
            train_state=ocp.args.StandardSave(item=items["train_state"]),
            config=ocp.args.JsonSave(item=items["config"]),
            metadata=ocp.args.JsonSave(item=metadata_dict),
        )

        # Save checkpoint with error handling (args carry the data in this API)
        self.manager.save(step, args=save_args)
        # In async mode, do not block here; wait at close().
        if not self.async_save:
            self.manager.wait_until_finished()
        return checkpoint_path_for_step(self.checkpoint_dir, step)

    def load_checkpoint(
        self,
        checkpoint_path: str | None = None,
        step: int | None = None,
        train_state_target: Any | None = None,
    ) -> tuple[Any, Any, CheckpointMetadata]:
        """Load checkpoint from Orbax format."""
        # Determine step either from explicit arg, from path, or via latest_step.
        if step is None:
            if checkpoint_path is not None:
                p = Path(checkpoint_path)
                name = p.name
                if name.startswith("checkpoint_"):
                    step = int(name.split("_")[-1])
                elif name.isdigit():
                    step = int(name)
                else:
                    # Fall back to latest if path is the root directory.
                    step = self.manager.latest_step()
                if step is None:
                    raise ValueError("No checkpoints found")
            else:
                step = self.manager.latest_step()
                if step is None:
                    raise ValueError("No checkpoints found")

        import jax
        from jax.sharding import SingleDeviceSharding

        # Saved host arrays and ShapeDtypeStruct targets have no sharding, so
        # Orbax restores them as NumPy arrays. The training dispatcher owns
        # subsequent device placement.
        orbax_train_state_target = (
            replace_zero_size_arrays(train_state_target) if train_state_target is not None else None
        )
        # Checkpoints saved as device arrays record their device. Orbax uses the
        # fallback only when that device is absent, e.g. a released checkpoint
        # saved from cpu:0 and restored under a GPU backend.
        fallback_sharding = SingleDeviceSharding(jax.local_devices()[0])

        restore_args = ocp.args.Composite(
            train_state=ocp.args.StandardRestore(
                item=orbax_train_state_target,
                strict=True,
                fallback_sharding=fallback_sharding,
            ),
            config=ocp.args.JsonRestore(item=None),
            metadata=ocp.args.JsonRestore(item=None),
        )

        checkpoint_data = self.manager.restore(step, args=restore_args)

        # Extract components (Composite result supports attribute-style access).
        train_state_data = restore_zero_size_arrays_from_target(checkpoint_data.train_state, train_state_target)
        config_data = checkpoint_data.config
        metadata_dict = getattr(checkpoint_data, "metadata", {}) or {}

        # Extract training config from config_data if not in metadata
        experiment_config = config_data.get("experiment", {}) if config_data else {}

        # Create metadata object
        metadata = CheckpointMetadata(
            step=metadata_dict.get("step", step),
            update_number=metadata_dict.get("update_number", 0),
            global_timestep=metadata_dict.get("global_timestep", 0),
            target_global_timestep=metadata_dict.get("target_global_timestep", 0),
            learning_rate=metadata_dict.get("learning_rate", 0.0),
            # Training config
            num_envs=metadata_dict.get("num_envs") or experiment_config.get("num_envs", 0),
            num_steps=metadata_dict.get("num_steps") or experiment_config.get("num_steps", 0),
            num_minibatches=metadata_dict.get("num_minibatches") or experiment_config.get("num_minibatches", 0),
            update_epochs=metadata_dict.get("update_epochs") or experiment_config.get("update_epochs", 0),
            schema_version=metadata_dict.get("schema_version", "2.0"),
            algo_version=metadata_dict.get("algo_version", "PPOJax_v1"),
            backend=metadata_dict.get("backend", "jax"),
            env_name=metadata_dict.get("env_name", ""),
            wandb_run_id=metadata_dict.get("wandb_run_id", ""),
        )

        return (config_data, train_state_data), metadata

    def list_checkpoints(self) -> list:
        """List all Orbax checkpoints."""
        return [str(self.checkpoint_dir / f"checkpoint_{s}") for s in self.manager.all_steps()]

    def get_latest_step(self) -> int | None:
        """Get the latest checkpoint step."""
        return self.manager.latest_step()

    def wait_until_finished(self):
        """Wait for all async checkpoint operations to complete."""
        self.manager.wait_until_finished()

    def close(self):
        """Close the checkpoint manager and clean up resources."""
        if hasattr(self, "manager"):
            # Ensure all operations complete and close cleanly.
            self.manager.wait_until_finished()
            self.manager.close()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensures cleanup."""
        self.close()
        return False

    def _extract_train_state(self, agent_state: Any) -> dict:
        """Extract training state from agent state."""
        if isinstance(agent_state, Mapping):
            return dict(agent_state)
        if hasattr(agent_state, "train_state"):
            ts = agent_state.train_state
            payload = {
                "params": ts.params,
                "opt_state": ts.opt_state,
                "step": ts.step,
                "run_stats": ts.run_stats if hasattr(ts, "run_stats") else {},
            }
            if hasattr(agent_state, "runtime_state"):
                payload["runtime_state"] = replace_zero_size_arrays(agent_state.runtime_state)
            return payload
        raise TypeError("checkpoint state must be a serialized mapping or expose train_state")

    def _extract_config(self, agent_conf: Any) -> dict:
        """Extract configuration from agent conf."""
        if hasattr(agent_conf, "config"):
            try:
                return OmegaConf.to_container(agent_conf.config, resolve=True, throw_on_missing=False)
            except Exception:
                pass
        return {"config": str(agent_conf)}


class UnifiedCheckpointManager:
    """Checkpoint manager that uses Orbax for saving and loading only."""

    def __init__(
        self,
        checkpoint_dir: str,
        format: str = "auto",
        max_to_keep: int = 5,
        prefer_orbax: bool = True,
        async_save: bool = True,
        save_interval_steps: int = 1,
        save_concurrent_gb: int = 4,
        restore_concurrent_gb: int = 4,
    ):
        # Note: format and prefer_orbax are kept for API compatibility but not used
        # since we always use Orbax for saving
        self.checkpoint_dir = Path(checkpoint_dir)
        self.max_to_keep = max_to_keep

        # Always use Orbax for saving if available
        if ORBAX_AVAILABLE:
            self.save_manager = OrbaxCheckpointManager(
                checkpoint_dir,
                max_to_keep=max_to_keep,
                save_interval_steps=save_interval_steps,
                async_save=async_save,
                save_concurrent_gb=save_concurrent_gb,
                restore_concurrent_gb=restore_concurrent_gb,
            )
            self.format = CheckpointFormat.ORBAX
        else:
            raise ImportError("Orbax is required for checkpoint saving. Install with: pip install orbax-checkpoint")

        # Legacy pickle checkpoints are not supported

    def save_checkpoint(
        self, step: int, agent_conf: Any, agent_state: Any, metadata: CheckpointMetadata | None = None
    ) -> str:
        """Save checkpoint using Orbax format only."""
        return self.save_manager.save_checkpoint(step, agent_conf, agent_state, metadata)

    def load_checkpoint(
        self,
        checkpoint_path: str | None = None,
        train_state_target: Any | None = None,
    ) -> tuple[Any, Any, CheckpointMetadata]:
        """Load checkpoint (Orbax only)."""
        return self.save_manager.load_checkpoint(
            checkpoint_path,
            train_state_target=train_state_target,
        )

    def list_checkpoints(self) -> list:
        """List all Orbax checkpoints."""
        return self.save_manager.list_checkpoints()

    def get_format(self) -> str:
        """Get the current checkpoint format."""
        return self.format

    def wait_until_finished(self):
        """Wait for async operations to complete."""
        if hasattr(self.save_manager, "wait_until_finished"):
            self.save_manager.wait_until_finished()

    def close(self):
        """Close the manager and clean up resources."""
        if hasattr(self.save_manager, "close"):
            self.save_manager.close()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensures cleanup."""
        self.close()
        return False


def create_checkpoint_manager(
    checkpoint_dir: str,
    format: str = "auto",
    max_to_keep: int = 5,
    async_save: bool = True,
    save_interval_steps: int = 1,
    save_concurrent_gb: int = 4,
    restore_concurrent_gb: int = 4,
    **kwargs,
) -> UnifiedCheckpointManager:
    """Factory function to create a checkpoint manager.

    Note: Always uses Orbax for saving and loading.
    The format parameter is kept for API compatibility.
    """
    return UnifiedCheckpointManager(
        checkpoint_dir=checkpoint_dir,
        format=format,
        max_to_keep=max_to_keep,
        async_save=async_save,
        save_interval_steps=save_interval_steps,
        save_concurrent_gb=save_concurrent_gb,
        restore_concurrent_gb=restore_concurrent_gb,
        **kwargs,
    )
