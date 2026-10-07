import pickle
from dataclasses import dataclass
from pathlib import Path

from flax import struct

from musclemimic.algorithms.common.optimizer import (
    linear_lr_schedule,
    warmup_cosine_lr_schedule,
)
from musclemimic.utils.metrics import MetricsHandler


@dataclass(frozen=True)
class AgentConfBase:
    """
    Abstract base class for *static* agent configuration.
    Any subclass must implement the serialize and from_dict methods.
    """

    def serialize(self):
        """Serialize the agent configuration."""
        raise NotImplementedError

    @classmethod
    def from_dict(cls, d):
        """Create an instance of the configuration class from a dictionary."""
        raise NotImplementedError


@struct.dataclass
class AgentStateBase:
    """
    Abstract base class for *dynamic* agent state.
    Any subclass must implement the serialize and from_dict methods.
    """

    def serialize(self):
        raise NotImplementedError

    @classmethod
    def from_dict(cls, d, agent_conf):
        raise NotImplementedError


class JaxRLAlgorithmBase:
    """
    Base class defining the interface for all JAX RL algorithms.
    """

    # these class attributes must be set by subclasses
    _agent_conf: AgentConfBase
    _agent_state: AgentStateBase

    # prefix for all saved agent files
    _saved_agent_suffix = ".pkl"

    @classmethod
    def init_agent_conf(cls, env, config):
        """
        Initialize the agent configuration.
        """
        raise NotImplementedError

    @classmethod
    def build_training(
        cls,
        env,
        agent_conf: AgentConfBase,
        metrics_handler: MetricsHandler | None = None,
        val_env=None,
    ):
        """Prepare the functions and static inputs needed for training."""
        raise NotImplementedError

    @classmethod
    def play_policy(cls, train_state, env, config, n_envs, n_steps=None, record=False, key=None):
        raise NotImplementedError

    @classmethod
    def save_agent(cls, path, agent_conf: AgentConfBase, agent_state: AgentStateBase):
        """Save the agent state to a file."""
        path = Path(path)
        path = path / (cls.__name__ + "_saved")
        path = path.with_suffix(cls._saved_agent_suffix)
        # serialize agent state
        serialized_state = cls.serialize(agent_conf, agent_state)
        # save agent state
        with open(path, "wb") as file:
            pickle.dump(serialized_state, file)
        print(f"\nSaved agent to: {path}\n")
        return path

    @classmethod
    def load_agent(cls, path):
        """Load the agent state from a file."""
        if isinstance(path, str):
            path = Path(path)
        if not path.is_file():
            raise ValueError(f"Not a file: {path}")
        if path.suffix != cls._saved_agent_suffix:
            raise ValueError(f"Not a {cls._saved_agent_suffix} file: {path}")
        with open(path, "rb") as file:
            data = pickle.load(file)
        return cls.from_dict(data)

    @classmethod
    def serialize(cls, agent_conf: AgentConfBase, agent_state: AgentStateBase):
        """Serialize conf and state of an agent."""
        serialized_agent_conf = agent_conf.serialize()
        serialized_agent_state = agent_state.serialize()
        return {"agent_conf": serialized_agent_conf, "agent_state": serialized_agent_state}

    @classmethod
    def from_dict(cls, d):
        """Load conf and state of an agent from a dictionary."""
        agent_conf = cls._agent_conf.from_dict(d["agent_conf"])
        agent_state = cls._agent_state.from_dict(d["agent_state"], agent_conf)
        return agent_conf, agent_state

    @staticmethod
    def _wrap_env(env, config):
        """Wrap the environment with the necessary wrappers."""
        raise NotImplementedError

    @classmethod
    def _linear_lr_schedule(cls, count, num_minibatches, update_epochs, lr, num_updates):
        return linear_lr_schedule(count, num_minibatches, update_epochs, lr, num_updates)

    @classmethod
    def _warmup_cosine_lr_schedule(
        cls,
        count,
        num_minibatches,
        update_epochs,
        lr,
        num_updates,
        warmup_steps=None,
        min_lr_ratio=0.0,
    ):
        return warmup_cosine_lr_schedule(
            count,
            num_minibatches,
            update_epochs,
            lr,
            num_updates,
            warmup_steps,
            min_lr_ratio,
        )

    @classmethod
    def build_runtime_state_restore_target(
        cls,
        rng,
        env,
        agent_conf,
        *,
        local_num_envs=None,
    ):
        """Build an algorithm-specific exact-resume runtime target."""
        raise NotImplementedError
