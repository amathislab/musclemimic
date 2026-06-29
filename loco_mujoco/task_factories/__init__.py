# ruff: noqa: I001

from .base import TaskFactory
from .rl_factory import RLFactory
from .imitation_factory import ImitationFactory
from .dataset_confs import AMASSDatasetConf, C3DDatasetConf, CustomDatasetConf, LAFAN1DatasetConf

# register factories
RLFactory.register()
ImitationFactory.register()
