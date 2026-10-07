from __future__ import annotations
from copy import deepcopy
from typing import List, Union
from collections import UserDict

import numpy as np
import mujoco
import jax.numpy as jnp
from dataclasses import make_dataclass
from flax import struct
from scipy.spatial.transform import Rotation as np_R
from jax.scipy.spatial.transform import Rotation as jnp_R

from loco_mujoco.core.stateful_object import StatefulObject
from loco_mujoco.core.utils.mujoco import (mj_jnt_name2id, mj_jntname2qposid, mj_jntname2qvelid,
                                           mj_jntid2qposid, mj_jntid2qvelid)
from loco_mujoco.core.utils.math import quat_scalarfirst2scalarlast, calculate_relative_site_quantities


class ObservationIndexContainer:
    """
    Container for indices of different non-stateful observation types, used to store indices
    related to observations within Mujoco data structures or observations
    created in an environment.
    """

    def __init__(self):

        # add attributes for the different observation types
        for obs_type in ObservationType.list_all_non_stateful():
            setattr(self, obs_type.__name__, [])

        self.concatenated_indices = None

    def convert_to_numpy(self):
        """
        Converts all list attributes of the class to NumPy arrays in place.
        """
        ind = []
        # Iterate through all attributes of the class
        for attr_name in vars(self):
            attr_value = getattr(self, attr_name)
            if attr_value is not None:
                ind += attr_value
            # Check if the attribute is a list before converting
            if isinstance(attr_value, list):
                setattr(self, attr_name, np.array(attr_value, dtype=int))

        # this array concatenates all indices in the order of this class
        self.concatenated_indices = np.argsort(np.array(ind))


class ObservationContainer(UserDict):
    """
    Container for observations. This is a dictionary with additional functionality to set the container reference
    for each observation.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._stateful_obs = []
        self._state_type = None
        self._locked = False

    def __setitem__(self, key, observation: Observation):

        if not self._locked:

            if observation.name in self.keys():
                raise KeyError("Duplicate keys are not allowed. Key: ", observation.name)

            # Set the container reference before adding to dict
            if isinstance(observation, Observation):
                observation.obs_container = self
            # Add the observation to the container
            super().__setitem__(key, observation)
            # Add the observation also to the stateful_obs container if it is a stateful observation
            if isinstance(observation, StatefulObservation):
                self._stateful_obs.append(observation)
        else:
            raise ValueError("Container is locked and cannot be modified.")

    def names(self):
        """Return a view of the dictionary's keys, same as keys()."""
        return self.keys()

    def entries(self):
        """Return a view of the dictionary's values, same as values()."""
        return self.values()

    def __eq__(self, other):
        if not isinstance(other, ObservationContainer):
            return False

        keys_are_the_same = set(self.keys()) == set(other.keys())
        type_self = [type(v) for v in self.values()]
        type_other = [type(v) for v in other.values()]
        types_are_the_same = type_self == type_other

        return keys_are_the_same and types_are_the_same

    def list_all_stateful(self):
        """
        Returns a list of all stateful observations in the container.
        """
        return self._stateful_obs

    def get_all_stateful_indices(self):
        """
        Get the indices of all stateful observations in the container.

        Returns:
            np.ndarray: The indices of the stateful observations.

        """
        return np.concatenate([obs.obs_ind for obs in self._stateful_obs]).astype(int)

    def init_state(self, env, key, model, data, backend):
        """
        Builds a dataclass from the stateful observations in the container.
        """

        # Get all stateful observations
        stateful_obs = self.list_all_stateful()
        # Create a dictionary with the stateful observations
        stateful_obs_dict = {obs.name: obs.init_state(env, key, model, data, backend) for obs in stateful_obs}
        if self._state_type is None:
            raise RuntimeError("ObservationContainer must be locked before initializing state.")
        return self._state_type(**stateful_obs_dict)

    def get_all_group_names(self):
        """
        Get all group names in the container.

        Returns:
            List[str]: The list of group names.

        """
        return list(set(group for obs in self.values() for group in obs.group))

    def filter_by_group(self, obs: Union[np.ndarray, jnp.ndarray], group_name: str = None):
        """
        Filter observations by group name.

        Args:
            obs (Union[np.ndarray, jnp.ndarray]): The observation array.
            group_name (str): The group name to filter by.

        Returns:
            Union[np.ndarray, jnp.ndarray]: The filtered observation array.

        """
        obs_ind_group = self.get_obs_ind_by_group(group_name)
        return obs[..., obs_ind_group]

    def get_obs_ind_by_group(self, group_name: str = None):
        """
        Get the indices of the observations by group name.

        Args:
            group_name (str): The group name to filter by.

        Returns:
            np.ndarray: The indices of the observations.

        """
        obs_ind_group = [obs.obs_ind for obs in self.values() if group_name in obs.group]
        return np.concatenate(obs_ind_group) if len(obs_ind_group) > 0 else np.array([])

    def get_randomizable_obs_indices(self):
        """
        Get the indices of the observations that are allowed to be randomized.

        Returns:
            np.ndarray: The indices of the observations.

        """
        return np.concatenate([obs.obs_ind for obs in self.values() if obs.allow_randomization]).astype(int)

    def reset_state(self, env, model, data, carry, backend):
        # Get all stateful observations
        stateful_obs = self.list_all_stateful()
        # Reset the state of each stateful observation
        for obs in stateful_obs:
             data, carry = obs.reset_state(env, model, data, carry, backend)
        return data, carry

    def lock(self):
        """
        Lock the container to prevent further modifications.
        """
        state_fields = [obs.name for obs in self._stateful_obs]
        state_type = make_dataclass("ObservationStates", state_fields)
        self._state_type = struct.dataclass(state_type, frozen=False)
        self._locked = True


class Observation:
    """
    Base class for all observation types.

    Args:
        obs_name: The name of the observation.
        group: The group name of the observation.
        allow_randomization: Whether the observation is allowed to be randomized.

    """

    registered = dict()

    def __init__(self, obs_name: str, group: Union[str, List(str)] = None, allow_randomization: bool = True):
        self.name = obs_name
        self.obs_container = None
        self.group = [group] if isinstance(group, str) or group is None else group
        self.allow_randomization = allow_randomization

        # these attributes *must* be initialized in the _init_from_mj method
        self.obs_ind = None
        self.data_type_ind = None
        self.min, self.max = None, None
        self._initialized_from_mj = False

    def init_from_mj(self, env, model, data, current_obs_size, data_ind_cont: ObservationIndexContainer,
                     obs_ind_cont: ObservationIndexContainer):
        """
        Initialize the observation type from the Mujoco data structure and model.

        Args:
            env: Environment instance.
            model: The Mujoco model.
            data: The Mujoco data structure.
            current_obs_size: The current size of the observation space.
            data_ind_cont (ObservationIndexContainer): The data indices container.
            obs_ind_cont (ObservationIndexContainer): The observation indices container.

        """
        # extract all information from data and model
        self._init_from_mj(env, model, data, current_obs_size)

        # store the indices in the ObservationIndexContainer
        self._add_to_data_and_obs_cont(data_ind_cont, obs_ind_cont)

    def _init_from_mj(self, env, model, data, current_obs_size):
        """
        Initialize the observation type from the Mujoco data structure and model.
        This method *must* initialize the following attributes:
            - obs_ind: Indices of this observation in the observation space.
            - data_type_ind: Indices of this observation in respective attribute
                in the Mujoco data structure (like qpos, qvel, xpos, etc.).
            - min: Minimum values of the observation in the observation space.
            - max: Maximum values of the observation in the observation space.

        Args:
            env: Environment instance.
            model: The Mujoco model.
            data: The Mujoco data structure.
            current_obs_size: The current size of the observation space.

        """
        raise NotImplementedError

    def init_from_traj(self, traj_handler):
        """
        Optionally, initialize the observation type to store relevant information from the trajectory.

        Args:
            traj_handler: Trajectory Handler class.

        """
        pass

    def _add_to_data_and_obs_cont(self, data_ind_cont: ObservationIndexContainer,
                                  obs_ind_cont: ObservationIndexContainer):
        """
        Adds the indices corresponding to this observation to the specified
        `ObservationIndexContainer` for both the MuJoCo data structure and the
        observation itself.

        Args:
            data_ind_cont (ObservationIndexContainer): The container holding
                the indices for the MuJoCo data structure.
            obs_ind_cont (ObservationIndexContainer): The container holding
                the indices for the observation.
        """

        # get the obs type name
        obs_type_name = self.__class__.__name__

        # store the indices in the ObservationIndexContainer
        data_ind_cont_attr = getattr(data_ind_cont, obs_type_name)
        obs_ind_cont_attr = getattr(obs_ind_cont, obs_type_name)
        data_ind_cont_attr.extend(deepcopy(self.data_type_ind.tolist()))
        obs_ind_cont_attr.extend(deepcopy(self.obs_ind.tolist()))

    @classmethod
    def get_all_obs_of_type(cls, env, model, data, data_ind_cont, backend):
        """
        Default getter for all the observations from the Mujoco data structure.

        Args:
            env: The environment.
            model: The Mujoco model.
            data: The Mujoco data structure.
            data_ind_cont (ObservationIndexContainer): The indices of *all* observations of all types.
            backend: The backend to use for the observation.

        Returns:
            The observation regarding this observation type.

        """
        obs_type_name = cls.__name__
        data_type_name = cls.data_type()

        assert data_type_name is not None, (f"Observation type {cls.__name__} does not have a default data_type. "
                                            f"Default get_obs method can not be used, please implement"
                                            f" a dedicated one.")

        # get the attribute from the mujoco data structure
        # For MJX data, some attributes (e.g., actuator_length, actuator_velocity)
        # should be accessed via data._impl to avoid deprecation warnings
        if hasattr(data, '_impl') and hasattr(data._impl, data_type_name):
            data_attr = getattr(data._impl, data_type_name)
        else:
            data_attr = getattr(data, data_type_name)
        # get the indices of all observations of this type
        data_ind = getattr(data_ind_cont, obs_type_name)
        # return the observation
        return backend.ravel(data_attr[data_ind])

    @classmethod
    def data_type(cls):
        """
        Attribute name in the mujoco data structure. If this is provided, the default get_obs method can be used.
        If not provided, a dedicated get_obs method must be implemented.

        Returns:
            The attribute name in the Mujoco data structure.
        """
        return None

    @property
    def initialized_from_mj(self):
        return self._initialized_from_mj

    @staticmethod
    def to_list(val):
        """
        Convert the input to a list of integers.
        """
        if isinstance(val, int):
            return [val]
        elif isinstance(val, np.ndarray) and val.dtype == int:
            return val.tolist()
        else:
            raise ValueError("Input must be an integer or a numpy array of integers")

    @classmethod
    def register(cls):
        """
        Register observation in the list and as an observation type.

        """
        obs_type_name = cls.__name__

        if obs_type_name not in Observation.registered:
            Observation.registered[obs_type_name] = cls

        # register it also in observation type class
        ObservationType.register(cls)

    @staticmethod
    def list_registered():
        """
        List registered Observations.

        Returns:
             The list of the registered goals.

        """
        return list(Observation.registered.keys())


class SimpleObs(Observation):
    """
    See also:
        :class:`Obs` for the base observation class.
    """

    def __init__(self, obs_name: str, xml_name: str, **kwargs):
        self.xml_name = xml_name
        super().__init__(obs_name, **kwargs)


class StatefulObservation(Observation, StatefulObject):

    def init_from_mj(self, env, model, data, current_obs_size, data_ind_cont: ObservationIndexContainer,
                     obs_ind_cont: ObservationIndexContainer):
        """
        Initialize the observation type from the Mujoco data structure and model.

        Args:
            env: Environment instance.
            model: The Mujoco model.
            data: The Mujoco data structure.
            current_obs_size: The current size of the observation space.
            data_ind_cont (ObservationIndexContainer): The data indices container.
            obs_ind_cont (ObservationIndexContainer): The observation indices container.

        """
        # extract all information from data and model
        self._init_from_mj(env, model, data, current_obs_size)

    @classmethod
    def get_all_obs_of_type(cls, env, model, data, data_ind_cont, backend):
        """ this function is not allowed to be called in this class. """
        raise NotImplementedError("Stateful observations do not support this function.")

    def get_obs_and_update_state(self, env, model, data, carry, backend):
        """
        Get the observation and update the state.

        Args:
            env: The environment.
            model: The Mujoco model.
            data: The Mujoco data structure.
            carry: The state carry.
            backend: The backend to use, either np or jnp.

        Returns:
            The observation and the updated state.

        """
        raise NotImplementedError


class BodyPos(SimpleObs):
    """
    Observation Type holding x, y, z position of the body.

    See also:
        :class:`Obs` for the base observation class.
    """

    dim = 3

    def _init_from_mj(self, env, model, data, current_obs_size):
        dim = len(data.body(self.xml_name).xpos)
        assert dim == self.dim
        self.min, self.max = [-np.inf] * dim, [np.inf] * dim
        self.data_type_ind = np.array(self.to_list(data.body(self.xml_name).id))
        self.obs_ind = np.array([j for j in range(current_obs_size, current_obs_size + dim)])
        self._initialized_from_mj = True

    @classmethod
    def data_type(cls):
        return "xpos"


class BodyRot(SimpleObs):
    """
    Observation Type holding the quaternion of the body.

    See also:
        :class:`Obs` for the base observation class.
    """

    dim = 4

    def _init_from_mj(self, env, model, data, current_obs_size):
        dim = len(data.body(self.xml_name).xquat)
        assert dim == self.dim
        self.min, self.max = [-np.inf] * dim, [np.inf] * dim
        self.data_type_ind = np.array(self.to_list(data.body(self.xml_name).id))
        self.obs_ind = np.array([j for j in range(current_obs_size, current_obs_size + dim)])
        self._initialized_from_mj = True

    @classmethod
    def data_type(cls):
        return "xquat"


class BodyVel(SimpleObs):
    """
    Observation Type holding the angular velocity around x, y, z and the linear velocity for x, y, z.

    See also:
        :class:`Obs` for the base observation class.
    """

    dim = 6

    def _init_from_mj(self, env, model, data, current_obs_size):
        dim = len(data.body(self.xml_name).cvel)
        assert dim == self.dim
        self.min, self.max = [-np.inf] * dim, [np.inf] * dim
        self.data_type_ind = np.array(self.to_list(data.body(self.xml_name).id))
        self.obs_ind = np.array([j for j in range(current_obs_size, current_obs_size + dim)])
        self._initialized_from_mj = True

    @classmethod
    def data_type(cls):
        return "cvel"


class FreeJointPos(SimpleObs):
    """
    Observation Type holding the 3D position and the 4D quaternion of a free joint.

    See also:
        :class:`Obs` for the base observation class.
    """

    dim = 7

    def _init_from_mj(self, env, model, data, current_obs_size):
        # note: free joints do not have limits
        self.min, self.max = [-np.inf] * FreeJointPos.dim, [np.inf] * FreeJointPos.dim
        self.data_type_ind = np.array(mj_jntname2qposid(self.xml_name, model))
        self.obs_ind = np.array([j for j in range(current_obs_size, current_obs_size + FreeJointPos.dim)])
        self._initialized_from_mj = True

    @classmethod
    def data_type(cls):
        return "qpos"


class EntryFromFreeJointPos(FreeJointPos):
    """
    Observation Type holding *a single entry* of a free joint pose.

    See also:
        :class:`Obs` for the base observation class.
    """

    dim = 1

    def __init__(self, entry_index: int, **kwargs):
        assert type(entry_index) == int, "entry_index must be an integer."
        self._entry_index = entry_index
        super().__init__(**kwargs)

    def _init_from_mj(self, env, model, data, current_obs_size):
        super()._init_from_mj(None, model, data, current_obs_size)
        self.min, self.max = [self.min[self._entry_index]], [self.max[self._entry_index]]
        self.data_type_ind = np.array([self.data_type_ind[self._entry_index]])
        self.obs_ind = np.array([self.obs_ind[0]])
        self._initialized_from_mj = True

    @classmethod
    def data_type(cls):
        return "qpos"


class FreeJointPosNoXY(FreeJointPos):
    """
    Observation Type holding the height and the 4D quaternion of a free joint pose, neglecting the x and y position.

    See also:
        :class:`Obs` for the base observation class.
    """

    dim = 5

    def _init_from_mj(self, env, model, data, current_obs_size):
        super()._init_from_mj(None, model, data, current_obs_size)
        self.min, self.max = self.min[2:], self.max[2:]
        self.data_type_ind = self.data_type_ind[2:]
        self.obs_ind = self.obs_ind[:-2]
        self._initialized_from_mj = True

    @classmethod
    def data_type(cls):
        return "qpos"


class JointPos(SimpleObs):
    """
    Observation Type holding the rotation of the joint.

    See also:
        :class:`Obs` for the base observation class.
    """

    dim = 1

    def _init_from_mj(self, env, model, data, current_obs_size):
        dim = len(data.joint(self.xml_name).qpos)
        assert dim == self.dim
        jh = model.joint(mj_jnt_name2id(self.xml_name, model))
        if jh.limited:
            self.min, self.max = [jh.range[0]], [jh.range[1]]
        else:
            self.min, self.max = [-np.inf] * dim, [np.inf] * dim
        self.data_type_ind = np.array(mj_jntid2qposid(jh.id, model))
        self.obs_ind = np.array([j for j in range(current_obs_size, current_obs_size + dim)])
        self._initialized_from_mj = True

    @classmethod
    def data_type(cls):
        return "qpos"


class JointPosArray(Observation):
    """
    Observation Type holding the rotation of an array of joints.

    See also:
        :class:`Obs` for the base observation class.
    """

    def __init__(self, obs_name: str, xml_names: List[str], **kwargs):
        super().__init__(obs_name, **kwargs)
        self._xml_names = xml_names
        self.dim = None

    def _init_from_mj(self, env, model, data, current_obs_size):
        dim = 0
        self.min, self.max, self.data_type_ind, self.obs_ind = [], [], [], []
        for name in self._xml_names:
            sdim = len(data.joint(name).qpos)
            jh = model.joint(mj_jnt_name2id(name, model))
            if jh.limited:
                min, max = [jh.range[0]] * sdim, [jh.range[1]] * sdim
            else:
                min, max = [-np.inf] * sdim, [np.inf] * sdim
            self.min.extend(min)
            self.max.extend(max)
            self.data_type_ind.extend(mj_jntid2qposid(jh.id, model))
            self.obs_ind.extend([j for j in range(current_obs_size, current_obs_size + sdim)])
            current_obs_size += sdim
            dim += sdim
        self.dim = dim
        self.data_type_ind = np.array(self.data_type_ind)
        self.obs_ind = np.array(self.obs_ind)
        self._initialized_from_mj = True

    @classmethod
    def data_type(cls):
        return "qpos"


class FreeJointVel(SimpleObs):
    """
    Observation Type holding the 3D linear velocity and the 3D angular velocity of a free joint.
    Note: Different to the BODY_VEL observation type!

    See also:
        :class:`Obs` for the base observation class.
    """

    dim = 6

    def _init_from_mj(self, env, model, data, current_obs_size):
        dim = len(data.joint(self.xml_name).qvel)
        # note: free joints do not have limits
        self.min, self.max = [-np.inf] * dim, [np.inf] * dim
        self.data_type_ind = np.array(mj_jntname2qvelid(self.xml_name, model))
        self.obs_ind = np.array([j for j in range(current_obs_size, current_obs_size + dim)])
        self._initialized_from_mj = True

    @classmethod
    def data_type(cls):
        return "qvel"


class EntryFromFreeJointVel(FreeJointVel):
    """
    Observation Type holding a single entry from of a free joint velocity.

    See also:
        :class:`Obs` for the base observation class.
    """
    dim = 1

    def __init__(self, entry_index: int, **kwargs):
        assert type(entry_index) == int, "entry_index must be an integer."
        self._entry_index = entry_index
        super().__init__(**kwargs)

    def _init_from_mj(self, env, model, data, current_obs_size):
        super()._init_from_mj(None, model, data, current_obs_size)
        self.min, self.max = [self.min[self._entry_index]], [self.max[self._entry_index]]
        self.data_type_ind = np.array([self.data_type_ind[self._entry_index]])
        self.obs_ind = np.array([self.obs_ind[0]])

    @classmethod
    def data_type(cls):
        return "qvel"


class JointVel(SimpleObs):
    """
    Observation Type holding the velocity of the joint.

    See also:
        :class:`Obs` for the base observation class.
    """

    dim = 1

    def _init_from_mj(self, env, model, data, current_obs_size):
        dim = len(data.joint(self.xml_name).qvel)
        assert dim == self.dim
        self.min, self.max = [-np.inf] * dim, [np.inf] * dim
        self.data_type_ind = np.array([model.jnt_dofadr[data.joint(self.xml_name).id]])
        self.obs_ind = np.array([j for j in range(current_obs_size, current_obs_size + dim)])
        self._initialized_from_mj = True

    @classmethod
    def data_type(cls):
        return "qvel"


class JointVelArray(Observation):
    """
    Observation Type holding the velocity of an array of joints.

    See also:
        :class:`Obs` for the base observation class.
    """

    def __init__(self, obs_name: str, xml_names: List[str], **kwargs):
        super().__init__(obs_name, **kwargs)
        self._xml_names = xml_names
        self.dim = None

    def _init_from_mj(self, env, model, data, current_obs_size):
        dim = 0
        self.min, self.max, self.data_type_ind, self.obs_ind = [], [], [], []
        for name in self._xml_names:
            sdim = len(data.joint(name).qvel)
            jh = model.joint(mj_jnt_name2id(name, model))
            min, max = [-np.inf] * sdim, [np.inf] * sdim
            self.min.extend(min)
            self.max.extend(max)
            self.data_type_ind.extend(mj_jntid2qvelid(jh.id, model))
            self.obs_ind.extend([j for j in range(current_obs_size, current_obs_size + sdim)])
            current_obs_size += sdim
            dim += sdim
        self.dim = dim
        self.data_type_ind = np.array(self.data_type_ind)
        self.obs_ind = np.array(self.obs_ind)
        self._initialized_from_mj = True

    @classmethod
    def data_type(cls):
        return "qvel"


class SitePos(SimpleObs):
    """
    Observation Type holding the x, y, z position of the site.

    See also:
        :class:`Obs` for the base observation class.
    """

    dim = 3

    def _init_from_mj(self, env, model, data, current_obs_size):
        dim = len(data.site(self.xml_name).xpos)
        assert dim == self.dim
        self.min, self.max = [-np.inf] * dim, [np.inf] * dim
        self.data_type_ind = np.array(self.to_list(data.site(self.xml_name).id))
        self.obs_ind = np.array([j for j in range(current_obs_size, current_obs_size + dim)])
        self._initialized_from_mj = True

    @classmethod
    def data_type(cls):
        return "site_xpos"


class SiteRot(SimpleObs):
    """
    Observation Type holding the flattened rotation matrix of the site.

    See also:
        :class:`Obs` for the base observation class.
    """

    dim = 9

    def _init_from_mj(self, env, model, data, current_obs_size):
        # Sites don't have rotation quaternion for some reason...
        # x_mat is rotation matrix with shape (9, )
        dim = len(data.site(self.xml_name).xmat)
        assert dim == self.dim
        self.min, self.max = [-np.inf] * dim, [np.inf] * dim
        self.data_type_ind = np.array(self.to_list(data.site(self.xml_name).id))
        self.obs_ind = np.array([j for j in range(current_obs_size, current_obs_size + dim)])
        self._initialized_from_mj = True

    @classmethod
    def data_type(cls):
        return "site_xmat"


class ProjectedGravityVector(Observation):
    """
    Observation Type holding the gravity vector.


    Args:
        obs_name: The name of the observation.
        xml_name: The name of the free joint in the Mujoco XML to calculate the gravity vector from.

    See also:
        :class:`Obs` for the base observation class.
    """

    dim = 3

    def __init__(self, obs_name: str, xml_name: str, **kwargs):
        self.xml_name = xml_name
        super().__init__(obs_name, **kwargs)

    def _init_from_mj(self, env, model, data, current_obs_size):
        self.min, self.max = [-np.inf] * self.dim, [np.inf] * self.dim
        self.data_type_ind = np.array(mj_jntname2qposid(self.xml_name, model))[3:]  # only the quaternion part is needed
        self.obs_ind = np.array([j for j in range(current_obs_size, current_obs_size + self.dim)])
        self._initialized_from_mj = True

    @classmethod
    def get_all_obs_of_type(cls, env, model, data, data_ind_cont, backend):
        """
        Default getter for all the projected gravity vectors from the Mujoco data structure.

        Args:
            env: The environment.
            model: The Mujoco model.
            data: The Mujoco data structure.
            data_ind_cont (ObservationIndexContainer): The indices of *all* observations of all types.
            backend: The backend to use for the observation.

        Returns:
            The observation regarding this observation type.

        """

        if backend == np:
            R = np_R
        elif backend == jnp:
            R = jnp_R
        else:
            raise ValueError(f"Unknown backend {backend}.")

        if data_ind_cont.ProjectedGravityVector.size == 0:
            return backend.empty(shape=(0,))
        else:
            xquats = data.qpos[data_ind_cont.ProjectedGravityVector]
            xquats = xquats.reshape(-1, 4)
            rots = R.from_quat(quat_scalarfirst2scalarlast(xquats))

            # get the gravity vector from the quaternions
            proj_grav = rots.inv().apply(np.array([0, 0, -1]))

            # return the observation
            return backend.ravel(proj_grav)


class Force(Observation):
    """
    Observation Type holding the collision forces/torques [3D force + 3D torque]
    between two geoms.

    See also:
        :class:`Obs` for the base observation class.
    """

    dim = 6

    def __init__(self, obs_name: str, xml_name_geom1: str, xml_name_geom2: str, **kwargs):
        self.xml_name_geom1 = xml_name_geom1
        self.xml_name_geom2 = xml_name_geom2
        super().__init__(obs_name, **kwargs)

        self.mjx_contact_id = None
        self.data_geom_id1 = None
        self.data_geom_id2 = None

    def _init_from_mj(self, env, model, data, current_obs_size):
        # get all required information from data
        self.min, self.max = [-np.inf] * self.dim, [np.inf] * self.dim
        self.data_geom_id1 = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, self.xml_name_geom1)
        self.data_geom_id2 = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, self.xml_name_geom2)
        self.data_type_ind = np.array([self.data_geom_id1, self.data_geom_id2])
        self.obs_ind = np.array([j for j in range(current_obs_size, current_obs_size + self.dim)])
        self._initialized_from_mj = True

    @classmethod
    def get_all_obs_of_type(cls, env, model, data, data_ind_cont, backend):
        ind = data_ind_cont.Force
        if backend == np:
            return backend.ravel(cls.mj_collision_force(model, data, ind))
        elif backend == jnp:
            return backend.ravel(cls.mjx_collision_force(model, data, ind))
        else:
            raise ValueError(f"Unknown backend {backend}.")

    @staticmethod
    def mj_collision_force(model, data, ind):

        c_array = np.zeros((len(ind), 6), dtype=np.float64)
        for i, geom_ids in enumerate(ind):

            for con_i in range(0, data.ncon):
                con = data.contact[con_i]
                con_geom_ids = (con.geom1, con.geom2)

                if geom_ids == con_geom_ids:
                    mujoco.mj_contactForce(model, data,
                                           con_i, c_array[i])

        return c_array

    @staticmethod
    def mjx_collision_force(model, data, ind):
        # will be added once mjx adds the collision force function to the official release
        c_array = np.zeros((len(ind), 6), dtype=np.float64)
        return c_array


class LastAction(StatefulObservation):

    def __init__(self, obs_name: str, **kwargs):
        super().__init__(obs_name, **kwargs)
        self.dim = None

    def _init_from_mj(self, env, model, data, current_obs_size):
        self.dim = env.action_dim
        self.min, self.max = [-np.inf] * self.dim, [np.inf] * self.dim
        self.obs_ind = np.array([j for j in range(current_obs_size, current_obs_size + self.dim)])
        self._initialized_from_mj = True

    def get_obs_and_update_state(self, env, model, data, carry, backend):
        """
        Get the observation and update the state.

        Args:
            env: The environment.
            model: The Mujoco model.
            data: The Mujoco data structure.
            carry: The state carry.
            backend: The backend to use, either np or jnp.

        Returns:
            The observation and the updated state.

        """
        return backend.ravel(carry.last_action), carry


class ModelInfo(StatefulObservation):

    def __init__(self, obs_name: str, model_attributes: Union[str, List[str]], **kwargs):
        self._model_attributes = [model_attributes] if isinstance(model_attributes, str) else model_attributes
        super().__init__(obs_name, **kwargs)
        self.dim = None

    def _init_from_mj(self, env, model, data, current_obs_size):
        dim = 0
        for attr in self._model_attributes:
            dim += getattr(model, attr).size
        self.dim = dim
        self.min, self.max = [-np.inf] * self.dim, [np.inf] * self.dim
        self.obs_ind = np.array([j for j in range(current_obs_size, current_obs_size + self.dim)])
        self._initialized_from_mj = True

    def get_obs_and_update_state(self, env, model, data, carry, backend):
        """
        Get the observation and update the state.

        Args:
            env: The environment.
            model: The Mujoco model.
            data: The Mujoco data structure.
            carry: The state carry.
            backend: The backend to use, either np or jnp.

        Returns:
            The observation and the updated state.

        """
        obs = []
        for attr in self._model_attributes:
            obs.append(backend.ravel(getattr(model, attr)))

        return backend.concatenate(obs), carry


class HeightMatrix(StatefulObservation):
    """
    Observation providing terrain height samples in a grid around the robot.
    Heights are relative to the robot's root body height.
    The grid rotates with the robot's heading (yaw) for proper terrain anticipation.

    Args:
        obs_name: Name of the observation.
        grid_rows: Number of rows in the sampling grid (default: 10).
        grid_cols: Number of columns in the sampling grid (default: 10).
        grid_resolution: Distance in meters between sample points (default: 0.1).
        body_name: Name of the body to center the grid on (default: root body).
    """

    def __init__(self, obs_name: str,
                 grid_rows: int = 10,
                 grid_cols: int = 10,
                 grid_resolution: float = 0.1,
                 body_name: str = None,
                 **kwargs):
        super().__init__(obs_name, **kwargs)
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols
        self.grid_resolution = grid_resolution
        self.body_name = body_name
        self.dim = grid_rows * grid_cols

        # To be initialized in _init_from_mj
        self._body_id = None
        self._grid_offsets_np = None   # Cached numpy array
        self._grid_offsets_jnp = None  # Cached JAX array
        self._free_jnt_qpos_id = None  # For extracting quaternion

    def _init_from_mj(self, env, model, data, current_obs_size):
        # Determine body to center the grid on
        if self.body_name is None:
            self.body_name = env.root_body_name
        self._body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, self.body_name)

        # Get free joint qpos indices for quaternion extraction
        self._free_jnt_qpos_id = np.array(mj_jntname2qposid(env.root_free_joint_xml_name, model))

        # Precompute grid offsets in LOCAL frame (forward=+x, lateral=+y)
        half_rows = (self.grid_rows - 1) / 2
        half_cols = (self.grid_cols - 1) / 2

        offsets = []
        for i in range(self.grid_rows):
            for j in range(self.grid_cols):
                x_offset = (i - half_rows) * self.grid_resolution  # Forward direction
                y_offset = (j - half_cols) * self.grid_resolution  # Lateral direction
                offsets.append([x_offset, y_offset])

        # Cache both numpy and JAX versions to avoid runtime conversion
        self._grid_offsets_np = np.array(offsets)
        self._grid_offsets_jnp = jnp.array(offsets)

        # Use unbounded range to avoid normalization/clipping issues
        # Actual height range depends on terrain elevation and robot position
        self.min = [-np.inf] * self.dim
        self.max = [np.inf] * self.dim
        self.obs_ind = np.array([j for j in range(current_obs_size, current_obs_size + self.dim)])
        self._initialized_from_mj = True

    def get_obs_and_update_state(self, env, model, data, carry, backend):
        """
        Get the height matrix observation and update the state.

        The grid is centered on the robot and rotated to align with the robot's
        heading direction. Heights are returned relative to the robot's current height.

        Args:
            env: The environment.
            model: The Mujoco model.
            data: The Mujoco data structure.
            carry: The state carry.
            backend: The backend to use, either np or jnp.

        Returns:
            Tuple of (observation array, updated carry).
        """
        # Get robot position
        body_pos = data.xpos[self._body_id]
        robot_x, robot_y, robot_z = body_pos[0], body_pos[1], body_pos[2]

        # Extract yaw from quaternion (qpos indices 3-6 for wxyz format)
        quat = data.qpos[self._free_jnt_qpos_id[3:7]]
        yaw = self._quat_to_yaw(quat, backend)

        # Get cached grid offsets for the appropriate backend
        if backend == jnp:
            grid_offsets = self._grid_offsets_jnp
        else:
            grid_offsets = self._grid_offsets_np

        # Rotate grid offsets to world frame using 2D rotation matrix
        cos_yaw = backend.cos(yaw)
        sin_yaw = backend.sin(yaw)

        # [cos -sin; sin cos] @ [x; y]
        rotated_x = cos_yaw * grid_offsets[:, 0] - sin_yaw * grid_offsets[:, 1]
        rotated_y = sin_yaw * grid_offsets[:, 0] + cos_yaw * grid_offsets[:, 1]

        # World coordinates of sample points
        sample_x = robot_x + rotated_x
        sample_y = robot_y + rotated_y

        # Sample terrain heights at these points
        heights = env._terrain.sample_heights_at_points(
            sample_x, sample_y, model, carry, backend
        )

        # Return heights relative to robot's current height
        relative_heights = heights - robot_z

        return backend.ravel(relative_heights), carry

    def _quat_to_yaw(self, quat, backend):
        """
        Extract yaw angle from quaternion in wxyz format.

        Args:
            quat: Quaternion array [w, x, y, z].
            backend: numpy or jax.numpy.

        Returns:
            Yaw angle in radians.
        """
        w, x, y, z = quat[0], quat[1], quat[2], quat[3]
        # Yaw = atan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        return backend.arctan2(siny_cosp, cosy_cosp)


class RelativeSiteQuantaties(StatefulObservation):
    """
    Observation Type holding the position, rotation and velocity of all sites for mimic relatively to the main site.
    This observation type is typically included in the observation space of imitation learning algorithms like
    AMP or DeepMimic, where the agent is supposed to mimic the trajectory of a reference agent.

    Args:
        obs_name: The name of the observation.
        site_names: List of site names to calculate the relative quantities from. The first site in the list is
            considered as the main site. If None, the environment's sites for mimic are used instead.

    See also:
        :class:`Obs` for the base observation class.
    """

    def __init__(self, obs_name: str, site_names: List[str] = None, **kwargs):
        self.site_names = site_names

        # will be initialized in the _init_from_mj method
        self.dim = None
        self.rel_site_ids = None
        self.site_bodyid = None
        self.body_rootid = None
        super().__init__(obs_name, **kwargs)

    def _init_from_mj(self, env, model, data, current_obs_size):
        if self.site_names is None:
            self.site_names = env.sites_for_mimic
            assert len(self.site_names) > 0, "No sites for mimic are defined in the environment."
        self.rel_site_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name) for name in self.site_names]
        self.rel_site_ids = np.array(self.rel_site_ids)
        self.site_bodyid = model.site_bodyid
        self.body_rootid = model.body_rootid
        n_sites = len(self.site_names) - 1
        self.dim = (3 + 3 + 6) * n_sites    # 3 for position, 3 for rotation, 6 for velocity
        self.min, self.max = [-np.inf] * self.dim, [np.inf] * self.dim
        self.obs_ind = np.array([j for j in range(current_obs_size, current_obs_size + self.dim)])
        self._initialized_from_mj = True

    def get_obs_and_update_state(self, env, model, data, carry, backend):
        """
        Get the observation and update the state.

        Args:
            env: The environment.
            model: The Mujoco model.
            data: The Mujoco data structure.
            carry: The state carry.
            backend: The backend to use, either np or jnp.

        Returns:
            The observation and the updated state.

        """

        rel_body_ids = self.site_bodyid[self.rel_site_ids]
        site_rpos, site_rangles, site_rvel = calculate_relative_site_quantities(data, self.rel_site_ids, rel_body_ids,
                                                                               self.body_rootid, backend)

        site_obs = backend.concatenate([
            backend.ravel(site_rpos),
            backend.ravel(site_rangles),
            backend.ravel(site_rvel)
        ])

        return backend.ravel(site_obs), carry


class ActuatorLength(SimpleObs):
    """
    Observation Type holding the length of an actuator (muscle).
    
    This observation type provides access to muscle/actuator lengths via MuJoCo's
    data.actuator_length field. This is particularly useful for muscle-driven models
    where muscle length information is important for control and analysis.
    
    See also:
        :class:`Observation` for the base observation class.
    """

    dim = 1

    def _init_from_mj(self, env, model, data, current_obs_size):
        """
        Initialize the actuator length observation from MuJoCo model.
        
        Args:
            env: The environment instance
            model: MuJoCo model
            data: MuJoCo data
            current_obs_size: Current observation size
        """
        # Find actuator ID by name
        actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, self.xml_name)
        self.data_type_ind = np.array([actuator_id])
        self.obs_ind = np.array([current_obs_size])
        
        # Set observation limits for muscle length
        # Muscle lengths typically range from 0 to a reasonable upper bound
        self.min = np.array([0.0])  # Minimum muscle length
        self.max = np.array([1.0])  # Maximum muscle length (TODO: find a reasonable upper bound)
        
        self._initialized_from_mj = True

    # Use default get_all_obs_of_type method from base class

    @classmethod
    def data_type(cls):
        """Return the data type for actuator length observations."""
        return "actuator_length"


class ActuatorVelocity(SimpleObs):
    """
    Observation Type holding the velocity of an actuator (muscle).
    
    This observation type provides access to muscle/actuator velocities via MuJoCo's
    data.actuator_velocity field. This is particularly useful for muscle-driven models
    where muscle velocity information is important for control and analysis.
    
    See also:
        :class:`Observation` for the base observation class.
    """

    dim = 1

    def _init_from_mj(self, env, model, data, current_obs_size):
        """
        Initialize the actuator velocity observation from MuJoCo model.
        
        Args:
            env: The environment instance
            model: MuJoCo model
            data: MuJoCo data
            current_obs_size: Current observation size
        """
        # Find actuator ID by name
        actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, self.xml_name)
        self.data_type_ind = np.array([actuator_id])
        self.obs_ind = np.array([current_obs_size])
        
        # Set observation limits for muscle velocity
        self.min = np.array([-np.inf])  # Muscle velocities can be negative
        self.max = np.array([np.inf])   # Muscle velocities can be positive
        
        self._initialized_from_mj = True

    @classmethod
    def data_type(cls):
        """Return the data type for actuator velocity observations."""
        return "actuator_velocity"


class ActuatorForce(SimpleObs):
    """
    Observation Type holding the force of an actuator (muscle).
    
    This observation type provides access to muscle/actuator forces via MuJoCo's
    data.actuator_force field. This is particularly useful for muscle-driven models
    where muscle force information is important for control and analysis.
    
    See also:
        :class:`Observation` for the base observation class.
    """

    dim = 1

    def _init_from_mj(self, env, model, data, current_obs_size):
        """
        Initialize the actuator force observation from MuJoCo model.
        
        Args:
            env: The environment instance
            model: MuJoCo model
            data: MuJoCo data
            current_obs_size: Current observation size
        """
        # Find actuator ID by name
        actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, self.xml_name)
        self.data_type_ind = np.array([actuator_id])
        self.obs_ind = np.array([current_obs_size])
        
        # Set observation limits for muscle force
        # Muscle forces are negative (muscles pull)
        # Force range is (-inf, 0]
        self.min = np.array([-np.inf])
        self.max = np.array([0.0])
        
        self._initialized_from_mj = True

    @classmethod
    def data_type(cls):
        """Return the data type for actuator force observations."""
        return "actuator_force"


class ActuatorExcitation(SimpleObs):
    """
    Observation Type holding the excitation (neural drive) of an actuator (muscle).

    This observation type provides access to muscle excitation via MuJoCo's
    data.ctrl field. This represents the neural command signal sent to muscles,
    before activation dynamics are applied.

    Note: For the actual muscle activation state after dynamics, use ActuatorActivation.

    See also:
        :class:`ActuatorActivation` for the actual activation state.
        :class:`Observation` for the base observation class.
    """

    dim = 1

    def _init_from_mj(self, env, model, data, current_obs_size):
        """
        Initialize the actuator excitation observation from MuJoCo model.

        Args:
            env: The environment instance
            model: MuJoCo model
            data: MuJoCo data
            current_obs_size: Current observation size
        """
        # Find actuator ID by name
        actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, self.xml_name)
        self.data_type_ind = np.array([actuator_id])
        self.obs_ind = np.array([current_obs_size])

        # Set observation limits for muscle excitation (control signal range)
        self.min = np.array([-1.0])     # Minimum excitation (we modified from default [0,1] to [-1,1])
                                        # check myofullbody.py for example
        self.max = np.array([1.0])      # Maximum excitation

        self._initialized_from_mj = True

    @classmethod
    def data_type(cls):
        """Return the data type for actuator excitation observations."""
        return "ctrl"


class ActuatorActivation(SimpleObs):
    """
    Observation Type holding the activation state of an actuator (muscle).

    This observation type provides access to the actual muscle activation state via
    MuJoCo's data.act field. This represents the contractile state of the muscle
    after first-order activation dynamics are applied.

    The activation state:
    - Lags behind the excitation (ctrl) due to activation/deactivation time constants
    - Is clamped to [0, 1] by MuJoCo (muscles can only pull, not push)
    - Directly determines the force output of the muscle

    Note: For the neural command signal, use ActuatorExcitation.

    See also:
        :class:`ActuatorExcitation` for the neural drive/command.
        :class:`Observation` for the base observation class.
    """

    dim = 1

    def _init_from_mj(self, env, model, data, current_obs_size):
        """
        Initialize the actuator activation observation from MuJoCo model.

        Args:
            env: The environment instance
            model: MuJoCo model
            data: MuJoCo data
            current_obs_size: Current observation size
        """
        # Find actuator ID by name
        actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, self.xml_name)
        self.data_type_ind = np.array([actuator_id])
        self.obs_ind = np.array([current_obs_size])

        # Set observation limits for muscle activation (always [0, 1] after dynamics)
        self.min = np.array([0.0])      # Minimum activation (muscles can't have negative activation)
        self.max = np.array([1.0])      # Maximum activation

        self._initialized_from_mj = True

    @classmethod
    def data_type(cls):
        """Return the data type for actuator activation observations."""
        return "act"


class TouchSensor(SimpleObs):
    """
    Observation Type holding the touch sensor value.
    
    This observation type provides access to touch sensor data via MuJoCo's
    data.sensordata field. Touch sensors provide scalar values indicating
    contact force magnitude, which are essential for locomotion and manipulation tasks.
    
    See also:
        :class:`Observation` for the base observation class.
    """
    
    dim = 1

    def _init_from_mj(self, env, model, data, current_obs_size):
        """
        Initialize the touch sensor observation from MuJoCo model.
        
        Args:
            env: The environment instance
            model: MuJoCo model
            data: MuJoCo data
            current_obs_size: Current observation size
        """
        # Get sensor ID from name
        sensor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, self.xml_name)
        if sensor_id == -1:
            raise ValueError(f"Sensor '{self.xml_name}' not found in model")
            
        # Verify this is a touch sensor
        sensor_type = model.sensor_type[sensor_id]
        if sensor_type != mujoco.mjtSensor.mjSENS_TOUCH:
            raise ValueError(f"Sensor '{self.xml_name}' is not a touch sensor (type: {sensor_type})")

        # Use sensor_adr to get the sensor index in sensordata
        self.data_type_ind = np.array([model.sensor_adr[sensor_id]])
        self.obs_ind = np.array([current_obs_size])
        
        # Touch sensors typically have values >= 0 (contact force magnitude)
        self.min = np.array([0.0])
        self.max = np.array([np.inf])
        
        self._initialized_from_mj = True

    @classmethod
    def data_type(cls):
        """Return the data type for touch sensor observations."""
        return "sensordata"


class ObservationType:
    """
    Namespace for all observation types for easy access.
    """
    BodyPos = BodyPos
    BodyRot = BodyRot
    BodyVel = BodyVel
    JointPos = JointPos
    JointVel = JointVel
    JointPosArray = JointPosArray
    JointVelArray = JointVelArray
    FreeJointPos = FreeJointPos
    EntryFromFreeJointPos = EntryFromFreeJointPos
    FreeJointPosNoXY = FreeJointPosNoXY
    FreeJointVel = FreeJointVel
    EntryFromFreeJointVel = EntryFromFreeJointVel
    SitePos = SitePos
    SiteRot = SiteRot
    ProjectedGravityVector = ProjectedGravityVector
    Force = Force
    LastAction = LastAction
    ModelInfo = ModelInfo
    RelativeSiteQuantaties = RelativeSiteQuantaties
    ActuatorLength = ActuatorLength
    ActuatorVelocity = ActuatorVelocity
    ActuatorForce = ActuatorForce
    ActuatorExcitation = ActuatorExcitation
    ActuatorActivation = ActuatorActivation
    TouchSensor = TouchSensor
    HeightMatrix = HeightMatrix

    @classmethod
    def get(cls, obs_name):
        """
        Get an observation type by name.

        Args:
            obs_name: The name of the observation type.

        Returns:
            The observation type.

        """
        return getattr(cls, obs_name)

    @classmethod
    def register(cls, new_obs_type):
        """
        Register a new observation type.

        Args:
            new_obs_type: The new observation type to register.

        """
        # Check if new_obs_type is a class
        if not isinstance(new_obs_type, type):
            raise TypeError(f"{new_obs_type} must be a class.")

        # Check if new_obs_type inherits from Observation
        if not issubclass(new_obs_type, Observation):
            raise TypeError(f"{new_obs_type.__name__} must inherit from Observation.")

        setattr(cls, new_obs_type.__name__, new_obs_type)

    @classmethod
    def list_all(cls):
        """
        List all observation types.
        """
        return [getattr(ObservationType, obs_type) for obs_type in dir(cls) if not obs_type.startswith("__")
                and obs_type not in ["register", "list_all", "list_all_non_stateful", "get"]]

    @classmethod
    def list_all_non_stateful(cls):
        """
        List all observation types, which are not stateful.
        """
        return [getattr(ObservationType, obs_type) for obs_type in dir(cls) if not obs_type.startswith("__")
                and obs_type not in ["register", "list_all", "list_all_non_stateful", "get"]
                and not issubclass(getattr(ObservationType, obs_type), StatefulObservation)]
