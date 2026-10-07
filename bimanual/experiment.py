import os

os.environ.setdefault(
    "XLA_FLAGS",
    "--xla_gpu_triton_gemm_any=True --xla_gpu_graph_min_graph_size=1",
)

import hydra
import jax
from omegaconf import DictConfig

from musclemimic.algorithms.common.distributed import device_count
from musclemimic.algorithms.common.trajectory_sharding import trajectory_sharding_mode
from musclemimic.runner.engine import run_experiment
from musclemimic.runner.logging import ExperimentHooks

jax.config.update("jax_default_matmul_precision", "high")


@hydra.main(version_base=None, config_path="./", config_name="conf_bimanual")
def experiment(config: DictConfig):
    n_used = device_count(config)
    if n_used > 1:
        print(
            f"[experiment] multi-GPU: n_used={n_used} / "
            f"n_visible={jax.local_device_count()}  sharding={trajectory_sharding_mode(config)}"
        )
    run_experiment(config, hooks=ExperimentHooks())


if __name__ == "__main__":
    experiment()
